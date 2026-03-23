import argparse
import glob
import os
import shutil
import sys

import pandas as pd


OLD_DIR_COLS = ["direction_right", "direction_left", "direction_up", "direction_down"]

EXPECTED_COLUMNS = [
    "episode_id",
    "pacman_x", "pacman_y",
    "moving_dir",
    "wall_right", "wall_left", "wall_up", "wall_down",
    "lives_remaining",
    "ghost_b_x", "ghost_b_y", "ghost_b_dist", "ghost_b_is_dangerous",
    "ghost_p_x", "ghost_p_y", "ghost_p_dist", "ghost_p_is_dangerous",
    "ghost_i_x", "ghost_i_y", "ghost_i_dist", "ghost_i_is_dangerous",
    "ghost_c_x", "ghost_c_y", "ghost_c_dist", "ghost_c_is_dangerous",
    "nearest_seed_dx", "nearest_seed_dy", "nearest_seed_dist",
    "seeds_right", "seeds_left", "seeds_down", "seeds_up",
    "nearest_energizer_dx", "nearest_energizer_dy", "nearest_energizer_dist",
    "energizers_remaining",
    "fruit_active",
    "seeds_eaten_ratio",
    "action",
]


def detect_format(df: pd.DataFrame) -> str:
    has_moving = "moving_dir" in df.columns
    has_dir = all(c in df.columns for c in OLD_DIR_COLS)

    if has_moving:
        return "v2"
    if has_dir:
        return "v1_full" if "is_reversing_right" in df.columns else "v1_dir"
    return "unknown"


def derive_moving_dir(df: pd.DataFrame) -> pd.Series:
    conditions = [
        df["direction_right"] == 1,
        df["direction_left"]  == 1,
        df["direction_up"]    == 1,
        df["direction_down"]  == 1,
    ]
    choices = [1, 2, 3, 4]
    return pd.Series(
        pd.array(
            [choices[next((i for i, c in enumerate(conditions) if c.iloc[row]), -1)]
             if any(c.iloc[row] for c in conditions) else 0
             for row in range(len(df))],
            dtype="int8",
        ),
        index=df.index,
        name="moving_dir",
    )


def migrate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["moving_dir"] = derive_moving_dir(df)

    drop_cols = [c for c in ["direction_right", "direction_left",
                              "direction_up", "direction_down",
                              "is_reversing_right", "is_reversing_left",
                              "is_reversing_up", "is_reversing_down"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    final_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[final_cols]


def migrate_file(src: str, dst: str) -> dict:
    df = pd.read_csv(src)
    fmt = detect_format(df)
    rows = len(df)

    if fmt == "v2":
        shutil.copy2(src, dst)
        return {"file": os.path.basename(src), "rows": rows,
                "status": "skipped (already v2 format)"}

    if fmt == "unknown":
        return {"file": os.path.basename(src), "rows": rows,
                "status": "skipped (unrecognised columns)"}

    migrated = migrate_dataframe(df)
    migrated.to_csv(dst, index=False)
    return {"file": os.path.basename(src), "rows": rows,
            "status": f"migrated ({fmt} → v2)"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Migrate Pacman recordings to v2 format: "
            "replaces direction_* one-hot columns with a single moving_dir integer.\n\n"
            "Encoding: 0=stopped, 1=right, 2=left, 3=up, 4=down"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", default="recordings",
        help="Input directory containing CSV files to migrate (default: recordings)",
    )
    parser.add_argument(
        "--output", default="recordings_v2",
        help=(
            "Output directory for migrated CSV files (default: recordings_v2). "
            "Ignored when --inplace is set."
        ),
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite original files in-place (a .bak backup is created for each).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Also search subdirectories of --input for CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pattern = (
        os.path.join(args.input, "**", "*.csv")
        if args.recursive
        else os.path.join(args.input, "*.csv")
    )
    files = sorted(glob.glob(pattern, recursive=args.recursive))
    if not files:
        sys.exit(f"[migrate-v2] No CSV files found under '{args.input}'.")

    print(f"[migrate-v2] Found {len(files)} CSV file(s) under '{args.input}'")
    print(f"[migrate-v2] Moving_dir encoding: 0=stopped 1=right 2=left 3=up 4=down\n")

    results = []
    for src in files:
        if args.inplace:
            bak = src + ".bak"
            shutil.copy2(src, bak)
            dst = src
        else:
            rel = os.path.relpath(src, args.input)
            dst = os.path.join(args.output, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

        info = migrate_file(src, dst)
        results.append(info)
        print(f"  [{info['status']}]  {info['file']}  ({info['rows']:,} rows)")

    migrated = sum(1 for r in results if r["status"].startswith("migrated"))
    skipped  = len(results) - migrated
    total_rows = sum(r["rows"] for r in results if r["status"].startswith("migrated"))

    print()
    print(
        f"[migrate-v2] Done.  {migrated} file(s) migrated ({total_rows:,} rows total), "
        f"{skipped} skipped."
    )
    if not args.inplace and migrated > 0:
        print(f"[migrate-v2] Output written to '{args.output}'")
        print(f"[migrate-v2] When satisfied, replace your recordings dir:")
        print(f"             mv {args.input} {args.input}_old && mv {args.output} {args.input}")


if __name__ == "__main__":
    main()
