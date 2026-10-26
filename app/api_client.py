"""Thin client for the KisanSense API. The frontend never imports model code."""
from __future__ import annotations

import os

import requests

API_URL = os.environ.get("KISANSENSE_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 90  # seconds; the first disease request loads TensorFlow and can be slow


class ApiError(Exception):
    """An error with a message that is safe to show to the user."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _detail_text(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200] or f"HTTP {response.status_code}"
    if isinstance(detail, list):  # pydantic validation errors
        parts = []
        for d in detail:
            field = ".".join(str(x) for x in d.get("loc", [])[1:]) or "input"
            parts.append(f"{field}: {d.get('msg', 'invalid value')}")
        return "; ".join(parts)
    return str(detail) if detail else f"HTTP {response.status_code}"


def _ensure_embedded() -> None:
    """In embedded mode (single-app hosts) start the API and fetch models on first use."""
    try:
        import embedded
    except ImportError:
        return
    try:
        embedded.ensure_ready()
    except Exception as e:  # download or start-up failed
        raise ApiError(f"The service could not start: {e}") from e


def _request(method: str, path: str, **kwargs):
    _ensure_embedded()
    try:
        r = requests.request(method, API_URL + path, timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError as e:
        raise ApiError(
            f"Cannot reach the KisanSense API at {API_URL}. Start it with: "
            "uvicorn backend.app.main:app"
        ) from e
    except requests.Timeout as e:
        raise ApiError("The API took too long to respond. Please try again.") from e
    if r.status_code >= 400:
        raise ApiError(_detail_text(r), r.status_code)
    return r.json()


def health() -> dict:
    return _request("GET", "/health")


def recommend_crop(payload: dict) -> dict:
    return _request("POST", "/api/v1/crop/recommend", json=payload)


def predict_disease(image_bytes: bytes, filename: str, content_type: str) -> dict:
    return _request("POST", "/api/v1/disease/predict", files={"file": (filename, image_bytes, content_type)})


def price_forecast(history_days: int = 90) -> dict:
    return _request("GET", "/api/v1/price/forecast", params={"history_days": history_days})


def metrics(module: str) -> dict | None:
    """Saved evaluation results, or None if that module has none yet."""
    try:
        return _request("GET", f"/api/v1/metrics/{module}")
    except ApiError as e:
        if e.status == 404:
            return None
        raise
