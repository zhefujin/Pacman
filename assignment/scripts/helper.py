import argparse
import glob
import os
import shutil
import sys

import pandas as pd


OLD_DIR_COLS = ["direction_right", "direction_left", "direction_up", "direction_down"]
NEW_REV_COLS = ["is_reversing_right", "is_reversing_left", "is_reversing_up", "is_reversing_down"]

INSERT_BEFORE = "wall_right"

EXPECTED_COLUMNS = [
    "episode_id",
    "pacman_x", "pacman_y",
    "direction_right", "direction_left", "direction_up", "direction_down",
    "is_reversing_right", "is_reversing_left", "is_reversing_up", "is_reversing_down",
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
    has_dir = all(c in df.columns for c in OLD_DIR_COLS)
    has_rev = all(c in df.columns for c in NEW_REV_COLS)
    if has_rev:
        return "new"
    if has_dir:
        return "old"
    return "unknown"


def migrate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["is_reversing_right"] = df["direction_left"].astype(int)
    df["is_reversing_left"]  = df["direction_right"].astype(int)
    df["is_reversing_up"]    = df["direction_down"].astype(int)
    df["is_reversing_down"]  = df["direction_up"].astype(int)
    final_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[final_cols]


def migrate_file(src: str, dst: str) -> dict:
    df = pd.read_csv(src)
    fmt = detect_format(df)
    rows = len(df)

    if fmt == "new":
        shutil.copy2(src, dst)
        return {"file": os.path.basename(src), "rows": rows, "status": "skipped (already new format)"}

    if fmt == "unknown":
        return {"file": os.path.basename(src), "rows": rows, "status": "skipped (unrecognised columns)"}

    migrated = migrate_dataframe(df)
    migrated.to_csv(dst, index=False)
    return {"file": os.path.basename(src), "rows": rows, "status": "migrated"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate Pacman recordings: add is_reversing_* columns derived from direction_*."
    )
    parser.add_argument(
        "--input", default="recordings",
        help="Input directory containing CSV files to migrate (default: recordings)"
    )
    parser.add_argument(
        "--output", default="recordings_migrated",
        help="Output directory for migrated CSV files (default: recordings_migrated). "
             "Ignored when --inplace is set."
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite original files in-place (a .bak backup is created for each file)."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Also search subdirectories of --input for CSV files."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.input, "**", "*.csv") if args.recursive \
              else os.path.join(args.input, "*.csv")
    files = sorted(glob.glob(pattern, recursive=args.recursive))
    if not files:
        pattern_sub = os.path.join(args.input, "*", "*.csv")
        files = sorted(glob.glob(pattern_sub))

    if not files:
        sys.exit(f"[migrate] No CSV files found under '{args.input}'.")

    print(f"[migrate] Found {len(files)} CSV file(s) under '{args.input}'")

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

        status = info["status"]
        rows   = info["rows"]
        fname  = info["file"]
        print(f"  [{status}]  {fname}  ({rows:,} rows)")
    migrated = sum(1 for r in results if r["status"] == "migrated")
    skipped  = len(results) - migrated
    total_rows = sum(r["rows"] for r in results if r["status"] == "migrated")
    print()
    print(f"[migrate] Done.  {migrated} file(s) migrated ({total_rows:,} rows total), "
          f"{skipped} skipped.")
    if not args.inplace:
        print(f"[migrate] Output written to '{args.output}'")
        print(f"[migrate] When satisfied, replace your recordings dir:")
        print(f"          mv {args.input} {args.input}_old && mv {args.output} {args.input}")


if __name__ == "__main__":
    main()
