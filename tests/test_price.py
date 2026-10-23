import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.price.backtest import rolling_backtest, score, split_origins  # noqa: E402
from kisansense.price.features import HORIZON, build_features, prepare_daily  # noqa: E402
from kisansense.price.models import DirectGBM, NaiveLast, SeasonalNaive7  # noqa: E402
from kisansense.price.predict import forecast, trend_signal  # noqa: E402


def _raw(n=500, drop_every=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 1200 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    df = pd.DataFrame({"date": dates, "modal_price": price, "arrival_qty": rng.uniform(300, 1200, n)})
    return df[df.index % drop_every != 5].reset_index(drop=True)


def test_prepare_daily_fills_short_gaps_and_flags_them():
    s = prepare_daily(_raw())
    assert s.index.is_monotonic_increasing
    assert (s.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert s["modal_price"].notna().all()
    assert s["was_missing"].sum() > 0


def test_prepare_daily_rejects_long_gap():
    df = _raw()
    df = df[(df["date"] < "2024-03-01") | (df["date"] > "2024-03-20")]
    with pytest.raises(ValueError):
        prepare_daily(df)


def test_seasonal_naive_uses_same_weekday_last_week():
    s = prepare_daily(_raw())
    hist = s.iloc[:200]
    pred = SeasonalNaive7().predict(hist)
    # forecast for day t+h equals the price at day t+h-7
    for h in range(1, HORIZON + 1):
        assert pred[h - 1] == hist["modal_price"].iloc[-7 + h - 1]


def test_naive_is_flat():
    s = prepare_daily(_raw())
    pred = NaiveLast().predict(s.iloc[:100])
    assert np.allclose(pred, s["modal_price"].iloc[99])


def test_features_do_not_use_the_future():
    s = prepare_daily(_raw())
    cut = 300
    f_full = build_features(s).iloc[cut]
    s_changed = s.copy()
    s_changed.iloc[cut + 1 :, :2] = 1.0  # destroy everything after the cut day
    f_changed = build_features(s_changed).iloc[cut]
    pd.testing.assert_series_equal(f_full, f_changed)


def test_gbm_prediction_ignores_data_after_origin():
    s = prepare_daily(_raw())
    hist = s.iloc[:320]
    m = DirectGBM().fit(hist)
    p1 = m.predict(hist)
    assert p1.shape == (HORIZON,) and (p1 > 0).all()


def test_backtest_shape_and_scores():
    s = prepare_daily(_raw(n=520))
    bt = rolling_backtest(s, model_names=["naive_last", "seasonal_naive_7"], initial_train=300, step=14)
    assert set(bt["model"]) == {"naive_last", "seasonal_naive_7"}
    assert set(bt["h"]) == set(range(1, HORIZON + 1))
    val, test = split_origins(bt)
    assert val["origin"].max() < test["origin"].min()
    sc = score(test)
    assert {"mae", "rmse", "mape_pct"} <= set(sc.columns)


def test_trend_signal_requires_move_larger_than_error():
    assert trend_signal(1000, [1010] * 7, 0.05)[0] == "no clear trend"
    assert trend_signal(1000, [1200] * 7, 0.05)[0] == "rising"
    assert trend_signal(1000, [800] * 7, 0.05)[0] == "falling"


def test_forecast_output_structure():
    s = prepare_daily(_raw())
    art = {
        "model": NaiveLast(),
        "model_name": "naive_last",
        "series": s,
        "interval_ratios": {h: (0.9, 1.1) for h in range(1, HORIZON + 1)},
        "trend_threshold": 0.05,
    }
    out = forecast(art)
    assert len(out["forecast"]) == len(out["low"]) == len(out["high"]) == HORIZON
    assert all(lo <= p <= hi for lo, p, hi in zip(out["low"], out["forecast"], out["high"]))
    assert out["trend"] == "no clear trend"
    assert "not selling advice" in out["disclaimer"]
