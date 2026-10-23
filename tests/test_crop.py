import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kisansense.common.config import CROP_FEATURE_RANGES, CROP_FEATURES, CROP_TARGET  # noqa: E402
from kisansense.crop.predict import InvalidInputError, recommend, validate_inputs  # noqa: E402
from kisansense.crop.train import build_models  # noqa: E402


def _synthetic_df(n_per_class: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for k, crop in enumerate(["rice", "wheat", "maize", "cotton"]):
        centre = {f: (lo + hi) * (0.2 + 0.2 * k) for f, (lo, hi) in CROP_FEATURE_RANGES.items()}
        for _ in range(n_per_class):
            r = {f: centre[f] * rng.uniform(0.9, 1.1) for f in CROP_FEATURES}
            r[CROP_TARGET] = crop
            rows.append(r)
    return pd.DataFrame(rows)

def _valid_input():
    return {f: (lo + hi) / 2 for f, (lo, hi) in CROP_FEATURE_RANGES.items()}

def test_validate_accepts_good_input():
    assert set(validate_inputs(_valid_input())) == set(CROP_FEATURES)

def test_validate_rejects_missing_value():
    bad = _valid_input()
    del bad["ph"]
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


def test_validate_rejects_out_of_range():
    bad = _valid_input()
    bad["ph"] = 99
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


def test_validate_rejects_non_numeric():
    bad = _valid_input()
    bad["N"] = "lots"
    with pytest.raises(InvalidInputError):
        validate_inputs(bad)


def test_recommend_returns_sorted_top3():
    df = _synthetic_df()
    model = build_models()["random_forest"].fit(df[CROP_FEATURES], df[CROP_TARGET])
    art = {"model": model, "features": CROP_FEATURES}
    out = recommend(_valid_input(), top_k=3, artifact=art)
    assert len(out) == 3
    probs = [o["probability"] for o in out]
    assert probs == sorted(probs, reverse=True)
    assert 0 <= sum(probs) <= 1.0 + 1e-9
