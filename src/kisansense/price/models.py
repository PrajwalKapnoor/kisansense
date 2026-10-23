"""Forecasters with a common interface.

    model = Model(); model.fit(history_df); prices = model.predict(history_df)

`history_df` is the prepared daily frame up to and including the forecast
origin. predict() returns HORIZON prices for the days after the last row.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from kisansense.common.config import RANDOM_SEED
from kisansense.price.features import HORIZON, build_features, horizon_frame


class NaiveLast:
    """Tomorrow looks like today. The baseline every model must beat."""

    name = "naive_last"

    def fit(self, hist):
        return self

    def predict(self, hist):
        return np.repeat(hist["modal_price"].iloc[-1], HORIZON).astype(float)


class SeasonalNaive7:
    """Each day looks like the same weekday last week."""

    name = "seasonal_naive_7"

    def fit(self, hist):
        return self

    def predict(self, hist):
        last7 = hist["modal_price"].iloc[-7:].to_numpy(dtype=float)
        # Day t+h maps to the value 7 days earlier: t+h-7, which is last7[h-1].
        return last7[:HORIZON].copy()


class MovingAverage7:
    name = "moving_average_7"

    def fit(self, hist):
        return self

    def predict(self, hist):
        return np.repeat(hist["modal_price"].iloc[-7:].mean(), HORIZON).astype(float)


class ARIMAModel:
    """ARIMA(1,1,1) on log price. Fixed order, not tuned per origin."""

    name = "arima"
    order = (1, 1, 1)

    def __init__(self):
        self._forecast = None
        self._fallback = None

    def fit(self, hist):
        from statsmodels.tsa.arima.model import ARIMA

        y = np.log(hist["modal_price"].astype(float))
        y.index = hist.index
        self._fallback = float(hist["modal_price"].iloc[-1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(y, order=self.order).fit()
                self._forecast = np.exp(np.asarray(res.forecast(HORIZON), dtype=float))
        except Exception:  # numerical failure: fall back to naive rather than crash a backtest
            self._forecast = None
        return self

    def predict(self, hist):
        if self._forecast is None:
            return np.repeat(self._fallback, HORIZON)
        return self._forecast.copy()


class DirectGBM:
    """One gradient-boosting model per horizon, predicting the log change from
    today's price using lag, rolling and arrival features."""

    name = "lag_gbm"

    def __init__(self):
        self.models = {}

    def fit(self, hist):
        feats = build_features(hist)
        lp = np.log(hist["modal_price"])
        for h in range(1, HORIZON + 1):
            X = horizon_frame(feats, h)
            y = lp.shift(-h) - lp  # known only if the target day is inside history
            ok = X.notna().all(axis=1) & y.notna()
            m = HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_depth=3,
                min_samples_leaf=20, random_state=RANDOM_SEED,
            )
            m.fit(X[ok], y[ok])
            self.models[h] = m
        return self

    def predict(self, hist):
        feats = build_features(hist).iloc[[-1]]
        last = float(hist["modal_price"].iloc[-1])
        out = []
        for h in range(1, HORIZON + 1):
            delta = self.models[h].predict(horizon_frame(feats, h))[0]
            out.append(last * float(np.exp(delta)))
        return np.array(out)


MODEL_FACTORIES = {
    "naive_last": NaiveLast,
    "seasonal_naive_7": SeasonalNaive7,
    "moving_average_7": MovingAverage7,
    "arima": ARIMAModel,
    "lag_gbm": DirectGBM,
}
