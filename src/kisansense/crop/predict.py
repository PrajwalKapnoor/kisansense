"""Crop recommendation inference with input validation."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from kisansense.common.config import CROP_FEATURE_RANGES, CROP_FEATURES
from kisansense.crop.train import MODEL_PATH


class InvalidInputError(ValueError):
    """Raised when a request has missing or out-of-range values."""


@lru_cache(maxsize=1)
def _load(path: str = str(MODEL_PATH)) -> dict:
    return joblib.load(path)


def validate_inputs(inputs: dict) -> dict:
    clean = {}
    for name in CROP_FEATURES:
        if name not in inputs:
            raise InvalidInputError(f"Missing value: {name}")
        try:
            value = float(inputs[name])
        except (TypeError, ValueError):
            raise InvalidInputError(f"{name} must be a number") from None
        lo, hi = CROP_FEATURE_RANGES[name]
        if not lo <= value <= hi:
            raise InvalidInputError(f"{name}={value} is outside the supported range {lo} to {hi}")
        clean[name] = value
    return clean


def recommend(inputs: dict, top_k: int = 3, artifact: dict | None = None) -> list[dict]:
    """Return the top_k crops with probabilities.

    Probabilities are the model's scores, not a guarantee of suitability.
    """
    clean = validate_inputs(inputs)
    art = artifact or _load()
    model = art["model"]
    row = pd.DataFrame([clean], columns=art["features"])
    proba = model.predict_proba(row)[0]
    order = proba.argsort()[::-1][:top_k]
    return [{"crop": str(model.classes_[i]), "probability": float(proba[i])} for i in order]
