"""Daily series preparation and lag features for price forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 7
LAGS = (1, 2, 3, 7, 14, 28)


def prepare_daily(df: pd.DataFrame, max_gap: int = 7) -> pd.DataFrame:
    """Reindex to a continuous daily calendar and interpolate short gaps.

    Adds `was_missing` so evaluation can ignore days whose value was filled in.
    Raises if a gap is longer than max_gap (we won't invent that much data).
    """
    s = df.set_index("date")[["modal_price", "arrival_qty"]].astype(float)
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full)
    s.index.name = "date"
    s["was_missing"] = s["modal_price"].isna()
    s[["modal_price", "arrival_qty"]] = s[["modal_price", "arrival_qty"]].interpolate(
        method="time", limit=max_gap, limit_direction="both"
    )
    if s[["modal_price", "arrival_qty"]].isna().any().any():
        raise ValueError(f"Found a gap longer than {max_gap} days; cannot build a continuous series")
    if (s["modal_price"] <= 0).any():
        raise ValueError("Non-positive prices found; clean the data before forecasting")
    return s


def build_features(s: pd.DataFrame) -> pd.DataFrame:
    """Features known at the end of each day (uses only that day and earlier)."""
    lp = np.log(s["modal_price"])
    la = np.log1p(s["arrival_qty"].clip(lower=0))
    f = pd.DataFrame(index=s.index)
    f["log_level"] = lp
    for k in LAGS:
        f[f"ret_{k}"] = lp - lp.shift(k)
    f["ma7_gap"] = np.log(s["modal_price"].rolling(7).mean()) - lp
    f["ma28_gap"] = np.log(s["modal_price"].rolling(28).mean()) - lp
    f["vol7"] = lp.diff().rolling(7).std()
    f["arrival_log"] = la
    f["arrival_vs_28"] = la - np.log1p(s["arrival_qty"].clip(lower=0).rolling(28).mean())
    return f


def horizon_frame(feats: pd.DataFrame, h: int) -> pd.DataFrame:
    """Add calendar features of the TARGET date (origin + h days)."""
    out = feats.copy()
    target_dates = out.index + pd.Timedelta(days=h)
    out["target_dow"] = target_dates.dayofweek
    out["target_month"] = target_dates.month
    return out
