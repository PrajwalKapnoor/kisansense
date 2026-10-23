import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.app import deps  # noqa: E402
from backend.app.main import app  # noqa: E402
from kisansense.common.config import CROP_FEATURE_RANGES, CROP_FEATURES, CROP_TARGET  # noqa: E402
from kisansense.crop.train import build_models  # noqa: E402
from kisansense.price.features import HORIZON, prepare_daily  # noqa: E402
from kisansense.price.models import NaiveLast  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clean_overrides():
    yield
    app.dependency_overrides.clear()


def _crop_artifact():
    rng = np.random.default_rng(0)
    rows = []
    for k, crop in enumerate(["rice", "wheat", "maize", "cotton"]):
        for _ in range(40):
            r = {f: (lo + hi) * (0.2 + 0.2 * k) * rng.uniform(0.9, 1.1) for f, (lo, hi) in CROP_FEATURE_RANGES.items()}
            r[CROP_TARGET] = crop
            rows.append(r)
    df = pd.DataFrame(rows)
    model = build_models()["random_forest"].fit(df[CROP_FEATURES], df[CROP_TARGET])
    return {"model": model, "features": CROP_FEATURES}


def _price_artifact():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    df = pd.DataFrame({"date": dates, "modal_price": 1200 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))),
                       "arrival_qty": rng.uniform(300, 1200, 300)})
    s = prepare_daily(df)
    return {"model": NaiveLast(), "model_name": "naive_last", "series": s,
            "interval_ratios": {h: (0.9, 1.1) for h in range(1, HORIZON + 1)}, "trend_threshold": 0.05}


def _valid_crop_body():
    return {f: (lo + hi) / 2 for f, (lo, hi) in CROP_FEATURE_RANGES.items()}


def _png_bytes(size=(80, 80)):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(3).integers(0, 255, (*size, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ---- health and metrics ----
def test_health_lists_models():
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["models"]) == {"crop", "price", "disease"}


def test_metrics_unknown_module_is_404():
    assert client.get("/api/v1/metrics/banana").status_code == 404


# ---- crop ----
def test_crop_recommend_ok():
    app.dependency_overrides[deps.crop_artifact] = _crop_artifact
    r = client.post("/api/v1/crop/recommend", json=_valid_crop_body())
    assert r.status_code == 200
    body = r.json()
    assert len(body["recommendations"]) == 3
    assert "agronomist" in body["note"]


def test_crop_out_of_range_is_422():
    app.dependency_overrides[deps.crop_artifact] = _crop_artifact
    bad = _valid_crop_body()
    bad["ph"] = 99
    assert client.post("/api/v1/crop/recommend", json=bad).status_code == 422


def test_crop_missing_field_is_422():
    app.dependency_overrides[deps.crop_artifact] = _crop_artifact
    bad = _valid_crop_body()
    del bad["rainfall"]
    assert client.post("/api/v1/crop/recommend", json=bad).status_code == 422


def test_crop_model_missing_is_503(monkeypatch):
    from kisansense.crop import predict

    def boom(*a, **k):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(predict, "_load", boom)
    r = client.post("/api/v1/crop/recommend", json=_valid_crop_body())
    assert r.status_code == 503
    assert "not available" in r.json()["detail"]


# ---- price ----
def test_price_forecast_ok():
    app.dependency_overrides[deps.price_artifact] = _price_artifact
    r = client.get("/api/v1/price/forecast?history_days=30")
    assert r.status_code == 200
    b = r.json()
    assert len(b["forecast"]) == len(b["low"]) == len(b["high"]) == HORIZON
    assert len(b["history"]["prices"]) == 30
    assert b["trend"] == "no clear trend"
    assert b["unit"] == "Rs/quintal"


def test_price_history_days_validated():
    app.dependency_overrides[deps.price_artifact] = _price_artifact
    assert client.get("/api/v1/price/forecast?history_days=2").status_code == 422


# ---- disease: upload validation (works without TensorFlow) ----
def _fake_disease_artifact():
    return ("model-placeholder", ["Tomato___healthy"])


def test_disease_rejects_wrong_content_type():
    app.dependency_overrides[deps.disease_artifact] = _fake_disease_artifact
    r = client.post("/api/v1/disease/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_disease_rejects_oversized_file():
    app.dependency_overrides[deps.disease_artifact] = _fake_disease_artifact
    big = b"\x00" * (5 * 1024 * 1024 + 10)
    r = client.post("/api/v1/disease/predict", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413


def test_disease_rejects_fake_image():
    app.dependency_overrides[deps.disease_artifact] = _fake_disease_artifact
    r = client.post("/api/v1/disease/predict", files={"file": ("a.png", b"not really a png", "image/png")})
    assert r.status_code == 400


def test_disease_model_missing_is_503(monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "DISEASE_FILE", tmp_path / "nope.keras")
    r = client.post("/api/v1/disease/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


# ---- disease: full prediction (needs TensorFlow) ----
def test_disease_predict_returns_gradcam():
    pytest.importorskip("tensorflow")
    from kisansense.disease.model import build_transfer

    model = build_transfer(3, (96, 96), weights=None)
    classes = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"]
    app.dependency_overrides[deps.disease_artifact] = lambda: (model, classes)
    r = client.post("/api/v1/disease/predict", files={"file": ("leaf.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    b = r.json()
    assert b["uncertain"] is True and b["prediction"] is None  # untrained model: near-uniform probabilities
    assert len(b["top"]) == 3 and set(b["top"][0]) == {"class", "name", "probability"}
    png = base64.b64decode(b["gradcam_png_base64"])
    assert Image.open(io.BytesIO(png)).size == (96, 96)
    assert "extension officer" in b["reminder"]
