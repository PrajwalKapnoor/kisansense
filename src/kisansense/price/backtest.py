"""Rolling-origin backtest: refit at each origin using only past data, forecast
the next HORIZON days, compare with what actually happened."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kisansense.price.features import HORIZON
from kisansense.price.models import MODEL_FACTORIES


def rolling_backtest(
    s: pd.DataFrame,
    model_names: list[str] | None = None,
    initial_train: int = 365,
    step: int = 7,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Return one row per (origin, model, horizon) with truth and prediction.

    Rows where the target day's price was interpolated are flagged so metrics
    can exclude them.
    """
    names = model_names or list(MODEL_FACTORIES)
    if len(s) < initial_train + horizon + step:
        raise ValueError("Not enough history for the requested backtest")

    price = s["modal_price"].to_numpy()
    missing = s["was_missing"].to_numpy()
    rows = []
    for i in range(initial_train - 1, len(s) - horizon, step):
        hist = s.iloc[: i + 1]  # only data up to the origin day
        for name in names:
            model = MODEL_FACTORIES[name]().fit(hist)
            pred = model.predict(hist)
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        "origin": hist.index[-1],
                        "model": name,
                        "h": h,
                        "y_true": float(price[i + h]),
                        "y_pred": float(pred[h - 1]),
                        "target_was_missing": bool(missing[i + h]),
                    }
                )
    return pd.DataFrame(rows)


def split_origins(bt: pd.DataFrame, val_fraction: float = 0.6):
    """Earlier origins for choosing a model, later origins for reporting."""
    origins = np.sort(bt["origin"].unique())
    cut = origins[int(len(origins) * val_fraction)]
    return bt[bt["origin"] < cut], bt[bt["origin"] >= cut]


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / MAPE per model on rows with real (not interpolated) targets."""
    d = bt[~bt["target_was_missing"]].copy()
    d["err"] = d["y_pred"] - d["y_true"]
    d["ape"] = d["err"].abs() / d["y_true"]
    g = d.groupby("model")
    out = pd.DataFrame(
        {
            "mae": g["err"].apply(lambda e: e.abs().mean()),
            "rmse": g["err"].apply(lambda e: float(np.sqrt((e**2).mean()))),
            "mape_pct": g["ape"].mean() * 100,
            "n": g.size(),
        }
    )
    return out.sort_values("mae")


def score_by_horizon(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt[~bt["target_was_missing"]].copy()
    d["abs_err"] = (d["y_pred"] - d["y_true"]).abs()
    return d.pivot_table(index="h", columns="model", values="abs_err", aggfunc="mean")
