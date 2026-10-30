"""Create the small summary files that the app's "Dataset and EDA" page shows (crop and price).

Run from the project root, with the raw data in place:
    python scripts/make_eda.py

Writes to docs/results/eda/:
    crop_eda.json     overview, class balance, feature distributions, per-crop medians, correlations
    price_eda.json    overview, gaps, seasonality, daily change distribution, arrivals vs price
    price_daily.csv   the cleaned daily Karnataka tomato series (date, modal_price, arrival_qty, was_missing)

Only summaries and the public daily price series are saved. The raw crop CSV is not copied.
The disease EDA files are created by the Colab notebook (cell 4b), because those images live in Colab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kisansense.common.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CROP_CSV,
    CROP_FEATURES,
    CROP_TARGET,
    DOCS_DIR,
    PRICE_CSV,
)
from kisansense.price.data import gap_report, load_price_data  # noqa: E402
from kisansense.price.features import prepare_daily  # noqa: E402

OUT = DOCS_DIR / "results" / "eda"


def _hist(values: pd.Series, bins: int = 20) -> dict:
    counts, edges = np.histogram(values.dropna(), bins=bins)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def crop_eda() -> dict:
    df = pd.read_csv(CROP_CSV)
    X, y = df[CROP_FEATURES], df[CROP_TARGET]
    counts = y.value_counts().sort_index()

    importance, importance_model = None, None
    model_file = ARTIFACTS_DIR / "crop_model.joblib"
    if model_file.exists():
        import joblib

        art = joblib.load(model_file)
        est = art["model"][-1] if hasattr(art["model"], "__getitem__") else art["model"]
        if hasattr(est, "feature_importances_"):
            importance = {f: round(float(v), 5) for f, v in zip(art["features"], est.feature_importances_)}
            importance_model = art.get("model_name")

    return {
        "overview": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "features": CROP_FEATURES,
            "classes": int(y.nunique()),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "rows_per_class_min": int(counts.min()),
            "rows_per_class_max": int(counts.max()),
        },
        "class_counts": {k: int(v) for k, v in counts.items()},
        "feature_stats": {
            f: {"min": float(X[f].min()), "max": float(X[f].max()), "mean": round(float(X[f].mean()), 3),
                "std": round(float(X[f].std()), 3)}
            for f in CROP_FEATURES
        },
        "histograms": {f: _hist(X[f]) for f in CROP_FEATURES},
        "crop_medians": {
            crop: {f: round(float(v), 3) for f, v in row.items()}
            for crop, row in df.groupby(CROP_TARGET)[CROP_FEATURES].median().iterrows()
        },
        "correlation": {
            "features": CROP_FEATURES,
            "matrix": [[round(float(v), 3) for v in row] for row in X.corr().to_numpy()],
        },
        "importance": importance,
        "importance_model": importance_model,
    }


def price_eda() -> tuple[dict, pd.DataFrame]:
    raw = load_price_data(PRICE_CSV)
    gaps = gap_report(raw)
    s = prepare_daily(raw)
    p = s["modal_price"]

    monthly = p.groupby(p.index.month).mean()
    ratio = p / p.rolling(7, center=True).mean()
    weekday = ratio.groupby(p.index.dayofweek).mean()
    log_ret = np.log(p).diff().dropna()
    change7 = (p.shift(-7) / p - 1).dropna() * 100

    la, lp = np.log1p(s["arrival_qty"].clip(lower=0)), np.log(p)
    quint = pd.qcut(s["arrival_qty"], 5, duplicates="drop")
    by_arrival = p.groupby(quint, observed=True).median()

    overview = {
        "start": str(p.index.min().date()),
        "end": str(p.index.max().date()),
        "days": int(len(p)),
        "present_days": int(gaps["present_days"]),
        "missing_days": int(gaps["missing_days"]),
        "longest_gap_days": int(gaps["longest_gap_days"]),
        "price_unit": str(raw["price_unit"].iloc[0]) if "price_unit" in raw.columns else "Rs/quintal",
        "min_price": float(p.min()),
        "max_price": float(p.max()),
        "max_price_date": str(p.idxmax().date()),
        "median_price": float(p.median()),
        "latest_price": float(p.iloc[-1]),
    }
    eda = {
        "overview": overview,
        "monthly_mean": {int(m): round(float(v), 1) for m, v in monthly.items()},
        "weekday_ratio": {int(d): round(float(v), 4) for d, v in weekday.items()},
        "daily_log_return_hist": _hist(log_ret, 30),
        "daily_log_return_std": round(float(log_ret.std()), 4),
        "change_7d_pct_quantiles": {str(q): round(float(change7.quantile(q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "arrival_price_corr": round(float(la.corr(lp)), 3),
        "median_price_by_arrival_quintile": [
            {"arrival_range": f"{iv.left:.0f} to {iv.right:.0f}", "median_price": round(float(v), 1)}
            for iv, v in by_arrival.items()
        ],
    }
    daily = s.reset_index()[["date", "modal_price", "arrival_qty", "was_missing"]]
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return eda, daily


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    crop = crop_eda()
    (OUT / "crop_eda.json").write_text(json.dumps(crop, indent=2))
    print(f"crop_eda.json   {crop['overview']['rows']} rows, {crop['overview']['classes']} crops")

    price, daily = price_eda()
    (OUT / "price_eda.json").write_text(json.dumps(price, indent=2))
    daily.to_csv(OUT / "price_daily.csv", index=False)
    print(f"price_eda.json  {price['overview']['days']} days ({price['overview']['start']} to {price['overview']['end']})")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
