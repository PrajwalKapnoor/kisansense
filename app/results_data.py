"""Load and reshape the saved training and EDA results for the results pages. No Streamlit in here."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FRACTION = 0.6  # same split of backtest origins that price/train.py uses

DISEASE_NAMES = {"baseline_cnn": "Small CNN (baseline)", "mobilenetv2_finetuned": "MobileNetV2, fine-tuned"}
CROP_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting (Hist)",
}
PRICE_NAMES = {
    "naive_last": "Naive (repeat last price)",
    "seasonal_naive_7": "Seasonal naive (7 days ago)",
    "moving_average_7": "7-day moving average",
    "arima": "ARIMA",
    "lag_gbm": "Gradient boosting on lags",
}


def results_dir() -> Path:
    return Path(os.environ.get("KISANSENSE_RESULTS_DIR", ROOT / "docs" / "results"))


def eda_dir() -> Path:
    return results_dir() / "eda"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def find_image(name: str) -> Path | None:
    for folder in (results_dir(), eda_dir()):
        p = folder / name
        if p.exists():
            return p
    return None


def pretty_class(name: str) -> str:
    s = name.replace("Tomato___", "").replace("_", " ")
    return s[:1].upper() + s[1:]


def pretty_crop(name: str) -> str:
    return name[:1].upper() + name[1:]


# ---------- disease ----------

def disease_table(m: dict) -> pd.DataFrame:
    rows = []
    for key, r in m["results"].items():
        rows.append(
            {
                "Model": DISEASE_NAMES.get(key, key) + ("  (selected)" if key == m["selected_model"] else ""),
                "Validation accuracy": r["validation"]["accuracy"],
                "Validation macro-F1": r["validation"]["macro_f1"],
                "Test accuracy": r["test"]["accuracy"],
                "Test macro-F1": r["test"]["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def disease_class_f1(m: dict) -> pd.DataFrame:
    best = m["results"][m["selected_model"]]["test"]["per_class_f1"]
    df = pd.DataFrame({"condition": [pretty_class(k) for k in best], "f1": list(best.values())})
    return df.sort_values("f1").reset_index(drop=True)


def class_balance(counts: dict) -> pd.DataFrame:
    df = pd.DataFrame({"condition": [pretty_class(k) for k in counts], "images": list(counts.values())})
    return df.sort_values("images", ascending=False).reset_index(drop=True)


TRAINING_PLAN = [
    ("Small CNN (baseline)", [("baseline_cnn", "Training from scratch")]),
    ("MobileNetV2, fine-tuned", [("transfer_head", "Phase 1: new head, base frozen"), ("transfer_finetune", "Phase 2: fine-tune top layers")]),
]


def training_frame(histories: dict) -> pd.DataFrame:
    """One row per model, epoch, split (Training/Validation) and metric (loss/accuracy)."""
    rows = []
    for model_label, phases in TRAINING_PLAN:
        epoch = 0
        for key, phase in phases:
            h = histories.get(key)
            if not h or not h.get("loss"):
                continue
            for i in range(len(h["loss"])):
                epoch += 1
                for split, prefix in (("Training", ""), ("Validation", "val_")):
                    for metric in ("loss", "accuracy"):
                        vals = h.get(prefix + metric)
                        if vals and i < len(vals):
                            rows.append(
                                {"model": model_label, "phase": phase, "epoch": epoch, "split": split,
                                 "metric": metric, "value": float(vals[i])}
                            )
    return pd.DataFrame(rows, columns=["model", "phase", "epoch", "split", "metric", "value"])


# ---------- crop ----------

def crop_table(m: dict) -> pd.DataFrame:
    rows = []
    for key, cv in m["cv"].items():
        t = m["test"][key]
        rows.append(
            {
                "Model": CROP_NAMES.get(key, key) + ("  (selected)" if key == m["selected_model"] else ""),
                "key": key,
                "CV macro-F1": cv["cv_f1_macro_mean"],
                "CV std": cv["cv_f1_macro_std"],
                "Test macro-F1": t["test_f1_macro"],
                "Test accuracy": t["test_accuracy"],
                "Test top-3 accuracy": t["test_top3_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def crop_class_f1(m: dict) -> pd.DataFrame:
    s = m["per_class_f1_selected"]
    df = pd.DataFrame({"crop": [pretty_crop(k) for k in s], "f1": list(s.values())})
    return df.sort_values(["f1", "crop"]).reset_index(drop=True)


def crop_medians_scaled(eda: dict) -> pd.DataFrame:
    """Per-crop median of each feature, with a 0-1 version (within each feature) for the colour scale."""
    rows = []
    feats = eda["overview"]["features"]
    for f in feats:
        vals = [eda["crop_medians"][c][f] for c in eda["crop_medians"]]
        lo, hi = min(vals), max(vals)
        for c in eda["crop_medians"]:
            v = eda["crop_medians"][c][f]
            rows.append({"crop": pretty_crop(c), "feature": f, "median": v, "scaled": 0.5 if hi == lo else (v - lo) / (hi - lo)})
    return pd.DataFrame(rows)


def correlation_long(eda: dict) -> pd.DataFrame:
    feats = eda["correlation"]["features"]
    rows = [
        {"a": a, "b": b, "r": eda["correlation"]["matrix"][i][j]}
        for i, a in enumerate(feats)
        for j, b in enumerate(feats)
    ]
    return pd.DataFrame(rows)


def strongest_correlation(eda: dict) -> tuple[str, str, float]:
    feats = eda["correlation"]["features"]
    best = (feats[0], feats[1], 0.0)
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            r = eda["correlation"]["matrix"][i][j]
            if abs(r) > abs(best[2]):
                best = (feats[i], feats[j], r)
    return best


def histogram_frame(h: dict) -> pd.DataFrame:
    edges = h["edges"]
    return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": h["counts"]})


# ---------- price ----------

def price_table(m: dict) -> pd.DataFrame:
    rows = []
    for key, t in m["test"].items():
        v = m["validation"].get(key, {})
        rows.append(
            {
                "Model": PRICE_NAMES.get(key, key) + ("  (selected)" if key == m["selected_model"] else ""),
                "key": key,
                "Validation MAE (Rs)": v.get("mae"),
                "Test MAE (Rs)": t["mae"],
                "Test RMSE (Rs)": t["rmse"],
                "Test MAPE (%)": t["mape_pct"],
                "Test MAE vs naive": m["test_mae_vs_naive"][key],
            }
        )
    return pd.DataFrame(rows)


def load_backtest(df: pd.DataFrame) -> pd.DataFrame:
    bt = df.copy()
    bt["origin"] = pd.to_datetime(bt["origin"])
    bt["target_date"] = bt["origin"] + pd.to_timedelta(bt["h"], unit="D")
    bt["target_was_missing"] = bt["target_was_missing"].astype(bool)
    return bt


def test_rows(bt: pd.DataFrame, val_fraction: float = VALIDATION_FRACTION) -> pd.DataFrame:
    origins = sorted(bt["origin"].unique())
    cut = origins[int(len(origins) * val_fraction)]
    return bt[bt["origin"] >= cut]


def mae_by_horizon(test: pd.DataFrame) -> pd.DataFrame:
    d = test[~test["target_was_missing"]].copy()
    d["abs_err"] = (d["y_pred"] - d["y_true"]).abs()
    out = d.groupby(["model", "h"], as_index=False)["abs_err"].mean().rename(columns={"abs_err": "mae"})
    out["model_name"] = out["model"].map(lambda k: PRICE_NAMES.get(k, k))
    return out


def forecast_vs_actual(test: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    d = test[(test["model"] == model) & (test["h"] == horizon)].sort_values("target_date")
    return d[["target_date", "y_true", "y_pred"]].reset_index(drop=True)


def pct_errors(test: pd.DataFrame, model: str) -> pd.Series:
    d = test[(test["model"] == model) & (~test["target_was_missing"])]
    return ((d["y_pred"] - d["y_true"]) / d["y_true"] * 100).reset_index(drop=True)
