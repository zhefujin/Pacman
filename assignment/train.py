import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


MODEL_FILENAME = "pacman_model.json"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

# Columns that are metadata / not used as features
NON_FEATURE_COLS = {"episode_id", "action"}

# "none" frames (Pacman not pressing any key) are the majority class and
# mostly uninformative — we drop them before training.
DROP_ACTIONS = {"none"}

# Minimum rows required to attempt training
MIN_ROWS = 500


def load_data(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        sys.exit(
            f"[train] No CSV files found in '{data_dir}'.\n"
            "Run the game with  python run.py --record  first."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if len(df) < 2:          # skip files that only have a header row
            print(f"  [skip] {path} — no data rows")
            continue
        df["source_file"] = os.path.basename(path)
        frames.append(df)
        print(f"  [load] {path}  ({len(df):,} rows, "
              f"{df['episode_id'].nunique()} episodes)")

    if not frames:
        sys.exit("[train] All CSV files were empty. Play more rounds first.")

    data = pd.concat(frames, ignore_index=True)
    print(f"\n[train] Total rows loaded: {len(data):,}")
    return data


def preprocess(data: pd.DataFrame):
    # Drop uninformative "none" action frames
    before = len(data)
    data = data[~data["action"].isin(DROP_ACTIONS)].copy()
    print(f"[train] Dropped {before - len(data):,} 'none'-action rows  "
          f"({len(data):,} remaining)")

    # Drop frames where the recorded action points directly into a wall.
    _wall_pairs = [
        ("right", "wall_right"),
        ("left",  "wall_left"),
        ("up",    "wall_up"),
        ("down",  "wall_down"),
    ]
    before_wall = len(data)
    into_wall_mask = pd.Series(False, index=data.index)
    for action, wall_col in _wall_pairs:
        into_wall_mask |= (data["action"] == action) & (data[wall_col] == 1)
    data = data[~into_wall_mask].copy()
    print(f"[train] Dropped {before_wall - len(data):,} into-wall frames  "
          f"({len(data):,} remaining)")

    if len(data) < MIN_ROWS:
        sys.exit(
            f"[train] Only {len(data)} usable rows after filtering — need at "
            f"least {MIN_ROWS}. Play more rounds and record again."
        )

    # Encode target labels  up/down/left/right -> 0/1/2/3
    le = LabelEncoder()
    y = le.fit_transform(data["action"])
    print(f"[train] Classes: {list(le.classes_)}")

    # Build feature matrix — drop metadata columns
    drop_cols = NON_FEATURE_COLS | {"source_file"}
    feature_cols = [c for c in data.columns if c not in drop_cols]
    X = data[feature_cols].values.astype(np.float32)

    groups = data["episode_id"].values

    _print_class_distribution(data["action"], le)

    return X, y, groups, le, feature_cols


def _print_class_distribution(action_series: pd.Series, le: LabelEncoder):
    counts = action_series.value_counts()
    total  = len(action_series)
    print("\n[train] Action distribution (training data):")
    for action, count in counts.items():
        bar = "█" * int(30 * count / total)
        print(f"  {action:<6}  {count:>6,}  ({100*count/total:4.1f}%)  {bar}")
    print()


def split_data(X, y, groups, test_size: float = 0.2, seed: int = 0):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    n_train_eps = len(set(groups[train_idx]))
    n_test_eps  = len(set(groups[test_idx]))
    print(f"[train] Split  →  train: {len(train_idx):,} frames "
          f"({n_train_eps} episodes)  |  "
          f"test: {len(test_idx):,} frames ({n_test_eps} episodes)")

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])


class CheckpointCallback(xgb.callback.TrainingCallback):
    def __init__(self, interval: int, checkpoint_dir: str):
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        # epoch is 0-indexed; epoch+1 is the human-readable round number
        if (epoch + 1) % self.interval == 0:
            path = os.path.join(
                self.checkpoint_dir, f"checkpoint_round_{epoch + 1:04d}.json"
            )
            model.save_model(path)
            print(f"[checkpoint] Round {epoch + 1:>4d} → {path}")
        return False  # returning True would stop training early


def build_model(num_classes: int, seed: int = 0,
                callbacks: list = None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        callbacks=callbacks or [],
        random_state=seed,
        n_jobs=-1,
    )


def train(X_train, X_test, y_train, y_test, num_classes: int,
          checkpoint_dir: str = "assignment/checkpoints",
          checkpoint_interval: int = 100):
    callbacks = [CheckpointCallback(checkpoint_interval, checkpoint_dir)]
    model = build_model(num_classes, callbacks=callbacks)
    print(f"[train] Fitting XGBoost  "
          f"(checkpoints every {checkpoint_interval} rounds → {checkpoint_dir}) …")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,          # print loss every 50 rounds
    )
    print(f"[train] Best iteration: {model.best_iteration}")
    return model


def evaluate(model, X_test, y_test, le: LabelEncoder):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[eval] Test accuracy: {acc*100:.2f}%\n")

    print("[eval] Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    _print_confusion_matrix(cm, le.classes_)

    return acc


def _print_confusion_matrix(cm: np.ndarray, class_names):
    col_w = 7
    header = " " * 8 + "".join(f"{c:>{col_w}}" for c in class_names)
    print("[eval] Confusion matrix (rows=true, cols=predicted):")
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {class_names[i]:<6}" + "".join(f"{v:>{col_w}}" for v in row)
        print(row_str)
    print()


def print_feature_importance(model, feature_cols: list, top_n: int = 15):
    scores = model.feature_importances_
    ranked = sorted(zip(feature_cols, scores), key=lambda t: t[1], reverse=True)
    print(f"[eval] Top {top_n} features by importance (gain):")
    max_score = ranked[0][1] if ranked else 1.0
    for name, score in ranked[:top_n]:
        bar = "█" * int(30 * score / max_score)
        print(f"  {name:<35}  {score:.4f}  {bar}")
    print()


def save_artifacts(model, le: LabelEncoder, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, MODEL_FILENAME)
    model.save_model(model_path)
    print(f"[save] Model  → {model_path}")

    le_path = os.path.join(model_dir, LABEL_ENCODER_FILENAME)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"[save] Encoder → {le_path}")


def load_artifacts(model_dir: str):
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    le_path = os.path.join(model_dir, LABEL_ENCODER_FILENAME)

    if not os.path.exists(model_path):
        sys.exit(f"[eval] Model not found at {model_path} — train first.")
    if not os.path.exists(le_path):
        sys.exit(f"[eval] Label encoder not found at {le_path} — train first.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return model, le


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on recorded Pacman gameplay."
    )
    parser.add_argument(
        "--data-dir", default="recordings",
        help="Directory containing recorded CSV files (default: recordings)"
    )
    parser.add_argument(
        "--model-dir", default="assignment",
        help="Base output directory (default: assignment). "
             "Model saved to <model-dir>/model/, checkpoints to <model-dir>/checkpoints/"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Save a checkpoint every N boosting rounds (default: 100)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of episodes held out for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load existing model and evaluate on the data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Pacman XGBoost Trainer")
    print("=" * 60)

    # Load & preprocess
    data = load_data(args.data_dir)
    X, y, groups, le, feature_cols = preprocess(data)

    # Episode-aware train/test split
    X_train, X_test, y_train, y_test = split_data(
        X, y, groups, test_size=args.test_size, seed=args.seed
    )

    model_dir = os.path.join(args.model_dir, "model")
    checkpoint_dir = os.path.join(args.model_dir, "checkpoints")

    if args.eval_only:
        print("\n[train] --eval-only: loading existing model …")
        model, le = load_artifacts(model_dir)
    else:
        model = train(
            X_train, X_test, y_train, y_test,
            num_classes=len(le.classes_),
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
        )
        save_artifacts(model, le, model_dir)

    # Evaluation
    evaluate(model, X_test, y_test, le)
    print_feature_importance(model, feature_cols)

    print("Done.")


if __name__ == "__main__":
    main()
