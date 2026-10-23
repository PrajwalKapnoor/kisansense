"""Model loading as FastAPI dependencies.

A missing or broken model file gives a clear 503 for that module only; the rest
of the API keeps working. TensorFlow is imported lazily so the API starts fast
and works without TensorFlow when only the crop and price modules are needed.
"""
from __future__ import annotations

from fastapi import HTTPException

from kisansense.common.config import ARTIFACTS_DIR

CROP_FILE = ARTIFACTS_DIR / "crop_model.joblib"
PRICE_FILE = ARTIFACTS_DIR / "price_model.joblib"
DISEASE_FILE = ARTIFACTS_DIR / "disease_model.keras"
DISEASE_CLASSES_FILE = ARTIFACTS_DIR / "disease_classes.json"


def _unavailable(module: str, why: str = "") -> HTTPException:
    detail = f"The {module} model is not available. Train it first and put its files in artifacts/."
    return HTTPException(status_code=503, detail=detail + (f" ({why})" if why else ""))


def crop_artifact():
    from kisansense.crop import predict

    try:
        return predict._load()
    except FileNotFoundError as e:
        raise _unavailable("crop recommendation") from e


def price_artifact():
    from kisansense.price import predict

    try:
        return predict._load()
    except FileNotFoundError as e:
        raise _unavailable("price forecasting") from e


def disease_artifact():
    if not (DISEASE_FILE.exists() and DISEASE_CLASSES_FILE.exists()):
        raise _unavailable("disease detection")
    try:
        # pyrefly: ignore[missing-import]
        from kisansense.disease import predict
    except ImportError as e:  # TensorFlow not installed
        raise _unavailable("disease detection", "TensorFlow is not installed") from e
    try:
        return predict.load()
    except Exception as e:  # corrupt or incompatible model file
        raise _unavailable("disease detection", "model file could not be loaded") from e
