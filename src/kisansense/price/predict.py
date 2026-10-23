"""Price forecast inference."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from kisansense.price.features import HORIZON
from kisansense.price.train import MODEL_PATH

DISCLAIMER = (
    "Experimental trend information from past Karnataka-wide prices. It ignores transport "
    "cost, quality, storage and local demand, and is not selling advice."
)


@lru_cache(maxsize=1)
def _load(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def trend_signal(last_price: float, forecast: list[float], threshold: float) -> tuple[str, float]:
    """Compare the average forecast with today's price.

    A move smaller than the model's typical 7-day error is reported as no clear
    trend instead of being dressed up as a signal.
    """
    change = (sum(forecast) / len(forecast)) / last_price - 1
    if abs(change) < threshold:
        return "no clear trend", change
    return ("rising" if change > 0 else "falling"), change


def forecast(artifact: dict | None = None, history_days: int = 90) -> dict:
    art = artifact or _load()
    s = art["series"]
    pred = [float(x) for x in art["model"].predict(s)]
    dates = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    ratios = art["interval_ratios"]
    low = [p * ratios[h][0] for h, p in enumerate(pred, start=1)]
    high = [p * ratios[h][1] for h, p in enumerate(pred, start=1)]
    last_price = float(s["modal_price"].iloc[-1])
    trend, change = trend_signal(last_price, pred, art["trend_threshold"])
    hist = s["modal_price"].iloc[-history_days:]
    return {
        "model": art["model_name"],
        "last_date": str(s.index[-1].date()),
        "last_price": last_price,
        "dates": [str(d.date()) for d in dates],
        "forecast": pred,
        "low": low,
        "high": high,
        "trend": trend,
        "expected_change_pct": change * 100,
        "trend_threshold_pct": art["trend_threshold"] * 100,
        "history": {"dates": [str(d.date()) for d in hist.index], "prices": [float(v) for v in hist.values]},
        "unit": "Rs/quintal",
        "disclaimer": DISCLAIMER,
    }
