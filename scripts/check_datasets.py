"""Day 1 data audit: checks that the three datasets exist and are usable.

Run from the project root:
    python scripts/check_datasets.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.common.config import (  # noqa: E402
    CROP_CSV,
    CROP_FEATURES,
    CROP_TARGET,
    DISEASE_DIR,
    PRICE_CSV,
)
from kisansense.price.data import gap_report, load_price_data  # noqa: E402

OK, WARN, FAIL = "[OK]  ", "[WARN]", "[FAIL]"


def check_crop() -> None:
    print("\n== Crop recommendation ==")
    if not CROP_CSV.exists():
        print(f"{FAIL} missing: {CROP_CSV}")
        return
    df = pd.read_csv(CROP_CSV)
    print(f"{OK} rows={len(df)}, cols={list(df.columns)}")
    missing_cols = [c for c in CROP_FEATURES + [CROP_TARGET] if c not in df.columns]
    if missing_cols:
        print(f"{FAIL} missing columns: {missing_cols}")
        return
    print(f"{OK} nulls: {int(df.isna().sum().sum())}, duplicate rows: {int(df.duplicated().sum())}")
    counts = df[CROP_TARGET].value_counts()
    print(f"{OK} classes={counts.size}, min/max per class={counts.min()}/{counts.max()}")
    group_like = [c for c in df.columns if c.lower() in {"state", "district", "region", "location"}]
    if group_like:
        print(f"{OK} possible group columns for grouped CV: {group_like}")
    else:
        print(f"{WARN} no region/group column: regional generalization cannot be evaluated; say so in the README")


def check_price() -> None:
    print("\n== Mandi prices ==")
    if not PRICE_CSV.exists():
        print(f"{FAIL} missing: {PRICE_CSV}")
        return
    try:
        df = load_price_data(PRICE_CSV)
    except ValueError as e:
        print(f"{FAIL} {e}")
        return
    print(f"{OK} rows={len(df)}, cols={list(df.columns)}")
    print(f"{OK} date range: {df['date'].min().date()} to {df['date'].max().date()}")
    for c in ("state", "commodity"):
        if c in df.columns:
            print(f"{OK} {c}: {sorted(df[c].dropna().unique())[:5]}")
    for c in ("arrival_unit", "price_unit"):
        if c in df.columns:
            print(f"{OK} {c} values: {sorted(df[c].dropna().unique())}")
    gaps = gap_report(df)
    level = OK if gaps["missing_days"] <= 0.05 * gaps["expected_days"] else WARN
    print(
        f"{level} days present={gaps['present_days']}/{gaps['expected_days']}, "
        f"missing={gaps['missing_days']}, longest gap={gaps['longest_gap_days']} days"
    )
    if "modal_price" in df.columns:
        p = df["modal_price"]
        print(f"{OK} modal_price: nulls={int(p.isna().sum())}, min={p.min():.0f}, median={p.median():.0f}, max={p.max():.0f}")
    if "arrival_qty" in df.columns:
        print(f"{OK} arrival_qty nulls={int(df['arrival_qty'].isna().sum())}")


def check_disease() -> None:
    print("\n== Disease images ==")
    if not DISEASE_DIR.exists():
        print(f"{FAIL} missing: {DISEASE_DIR}")
        return
    class_dirs = [d for d in DISEASE_DIR.rglob("*") if d.is_dir()]
    leaf_dirs = [d for d in class_dirs if any(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in d.iterdir())]
    if not leaf_dirs:
        print(f"{FAIL} no image folders found under {DISEASE_DIR}")
        return
    print(f"{OK} class folders with images: {len(leaf_dirs)}")
    for d in sorted(leaf_dirs)[:10]:
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"      {d.name}: {n}")
    if len(leaf_dirs) > 10:
        print(f"      ... and {len(leaf_dirs) - 10} more")


if __name__ == "__main__":
    check_crop()
    check_price()
    check_disease()