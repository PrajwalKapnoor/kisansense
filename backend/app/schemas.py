"""Request and response models. Crop input ranges come from the same config the
model code validates against, so the API and the model can never disagree."""
from __future__ import annotations

from pydantic import BaseModel, Field

from kisansense.common.config import CROP_FEATURE_RANGES as R


def _f(name: str, description: str):
    lo, hi = R[name]
    return Field(ge=lo, le=hi, description=f"{description} (allowed {lo} to {hi})")


class CropRequest(BaseModel):
    N: float = _f("N", "Nitrogen content in soil")
    P: float = _f("P", "Phosphorus content in soil")
    K: float = _f("K", "Potassium content in soil")
    temperature: float = _f("temperature", "Temperature in degrees Celsius")
    humidity: float = _f("humidity", "Relative humidity in percent")
    ph: float = _f("ph", "Soil pH")
    rainfall: float = _f("rainfall", "Rainfall in mm")


class CropRecommendation(BaseModel):
    crop: str
    probability: float


class CropResponse(BaseModel):
    recommendations: list[CropRecommendation]
    note: str


class TopClass(BaseModel):
    cls: str = Field(alias="class")
    name: str
    probability: float

    model_config = {"populate_by_name": True}


class DiseaseResponse(BaseModel):
    prediction: str | None
    name: str
    description: str
    confidence: float
    uncertain: bool
    top: list[TopClass]
    reminder: str
    gradcam_png_base64: str | None = None


class PriceHistory(BaseModel):
    dates: list[str]
    prices: list[float]


class PriceResponse(BaseModel):
    model: str
    last_date: str
    last_price: float
    dates: list[str]
    forecast: list[float]
    low: list[float]
    high: list[float]
    trend: str
    expected_change_pct: float
    trend_threshold_pct: float
    history: PriceHistory
    unit: str
    disclaimer: str
