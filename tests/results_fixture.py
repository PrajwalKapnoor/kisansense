"""Build a small, fake-but-realistic docs/results folder for testing the results pages.

The three metrics files are real results copied from the project; the curves, backtest rows, EDA files and
images are generated here so tests do not need data, a GPU or trained models.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
PRICE_MODELS = ["naive_last", "seasonal_naive_7", "moving_average_7", "arima", "lag_gbm"]
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROPS = ["rice", "maize", "jute", "cotton", "coffee", "mango"]


def _png(path: Path, text: str, size=(640, 360)) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def _histories() -> dict:
    rng = np.random.default_rng(0)

    def run(n, a0, a1, l0, l1):
        acc = np.linspace(a0, a1, n) + rng.normal(0, 0.01, n)
        vacc = np.linspace(a0 + 0.03, a1 - 0.02, n) + rng.normal(0, 0.01, n)
        loss = np.linspace(l0, l1, n)
        vloss = np.linspace(l0 * 0.9, l1 * 1.3, n)
        return {"accuracy": acc.tolist(), "val_accuracy": vacc.tolist(), "loss": loss.tolist(), "val_loss": vloss.tolist()}

    return {
        "baseline_cnn": run(13, 0.45, 0.88, 1.4, 0.35),
        "transfer_head": run(8, 0.70, 0.90, 0.9, 0.3),
        "transfer_finetune": run(6, 0.91, 0.97, 0.28, 0.1),
    }


def _backtest() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2024-01-01")
    price = 1200 + 250 * np.sin(np.arange(1100) / 60) + np.cumsum(rng.normal(0, 4, 1100))
    rows = []
    for k in range(91):
        i = 365 + 7 * k
        for model in PRICE_MODELS:
            noise = {"naive_last": 0.13, "seasonal_naive_7": 0.16, "moving_average_7": 0.13, "arima": 0.12, "lag_gbm": 0.12}[model]
            for h in range(1, 8):
                y = float(price[i + h])
                rows.append(
                    {"origin": (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "model": model, "h": h,
                     "y_true": y, "y_pred": y * (1 + rng.normal(0, noise * np.sqrt(h) / 2.2)),
                     "target_was_missing": bool(rng.random() < 0.015)}
                )
    return pd.DataFrame(rows)


def _hist(values, bins=20):
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def _crop_eda() -> dict:
    rng = np.random.default_rng(2)
    meds = {c: {f: round(float(rng.uniform(10, 120)), 2) for f in FEATURES} for c in CROPS}
    corr = np.eye(len(FEATURES))
    corr[1, 2] = corr[2, 1] = 0.74
    corr[4, 6] = corr[6, 4] = -0.31
    return {
        "overview": {"rows": 2200, "columns": 8, "features": FEATURES, "classes": len(CROPS), "missing_cells": 0,
                     "duplicate_rows": 0, "rows_per_class_min": 100, "rows_per_class_max": 100},
        "class_counts": {c: 100 for c in CROPS},
        "feature_stats": {f: {"min": 0.0, "max": 100.0, "mean": 50.0, "std": 20.0} for f in FEATURES},
        "histograms": {f: _hist(rng.normal(50, 20, 500)) for f in FEATURES},
        "crop_medians": meds,
        "correlation": {"features": FEATURES, "matrix": corr.round(3).tolist()},
        "importance": {f: round(float(v), 4) for f, v in zip(FEATURES, rng.dirichlet(np.ones(7)))},
        "importance_model": "random_forest",
    }


def _price_eda() -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-01-01", periods=1005, freq="D")
    price = 1100 + 300 * np.sin(np.arange(1005) / 70) + rng.normal(0, 70, 1005)
    daily = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "modal_price": price.round(0), "arrival_qty": rng.uniform(300, 1500, 1005).round(0),
                          "was_missing": False})
    eda = {
        "overview": {"start": "2024-01-01", "end": "2026-10-01", "days": 1005, "present_days": 989, "missing_days": 16,
                     "longest_gap_days": 6, "price_unit": "Rs/Quintal", "min_price": 520.0, "max_price": 4980.0,
                     "max_price_date": "2025-11-24", "median_price": 1080.0, "latest_price": 1627.0},
        "monthly_mean": {m: float(1000 + 40 * m) for m in range(1, 13)},
        "weekday_ratio": {d: float(1 + 0.004 * (d - 3)) for d in range(7)},
        "daily_log_return_hist": _hist(rng.normal(0, 0.05, 1000), 30),
        "daily_log_return_std": 0.05,
        "change_7d_pct_quantiles": {"0.1": -19.0, "0.25": -8.0, "0.5": 0.0, "0.75": 9.0, "0.9": 22.0},
        "arrival_price_corr": -0.12,
        "median_price_by_arrival_quintile": [{"arrival_range": f"{a} to {a + 200}", "median_price": 1000.0 + 20 * k}
                                             for k, a in enumerate(range(300, 1300, 200))],
    }
    return eda, daily


def build(dest: Path) -> Path:
    dest = Path(dest)
    (dest / "eda").mkdir(parents=True, exist_ok=True)
    for name in ("disease_metrics.json", "crop_metrics.json", "price_metrics.json"):
        shutil.copy2(FIXTURES / name, dest / name)
    (dest / "disease_histories.json").write_text(json.dumps(_histories()))
    _backtest().to_csv(dest / "price_backtest_rows.csv", index=False)
    for name in ("disease_confusion_mobilenetv2_finetuned.png", "disease_gradcam_examples.png", "crop_confusion_matrix.png",
                 "eda/disease_samples.png", "eda/disease_errors.png"):
        _png(dest / name, name)
    (dest / "eda" / "disease_eda.json").write_text(json.dumps({
        "images_total": 9350, "images_sampled": 600,
        "size_counts": [{"size": "256x256", "count": 600}],
        "brightness_by_class": {"Bacterial spot": {"mean": 0.41, "std": 0.07}, "Early blight": {"mean": 0.44, "std": 0.08},
                                "Late blight": {"mean": 0.39, "std": 0.07}, "Leaf Mold": {"mean": 0.46, "std": 0.09},
                                "Septoria leaf spot": {"mean": 0.42, "std": 0.07}, "healthy": {"mean": 0.43, "std": 0.06}}}))
    (dest / "eda" / "disease_errors.json").write_text(json.dumps({
        "mistakes": 122, "test_images": 1873,
        "top_confusions": [{"true": "Early blight", "predicted": "Septoria leaf spot", "count": 14},
                           {"true": "Late blight", "predicted": "Early blight", "count": 11}],
        "mean_confidence_when_wrong": 0.71, "mean_confidence_when_right": 0.96}))
    (dest / "eda" / "crop_eda.json").write_text(json.dumps(_crop_eda()))
    eda, daily = _price_eda()
    (dest / "eda" / "price_eda.json").write_text(json.dumps(eda))
    daily.to_csv(dest / "eda" / "price_daily.csv", index=False)
    return dest
