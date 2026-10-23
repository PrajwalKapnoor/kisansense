from fastapi import APIRouter, Depends, Query

from backend.app.deps import price_artifact
from backend.app.schemas import PriceResponse

router = APIRouter()


@router.get("/forecast", response_model=PriceResponse)
def price_forecast(
    history_days: int = Query(90, ge=7, le=365, description="Days of past prices to include for charting"),
    artifact=Depends(price_artifact),
):
    from kisansense.price.predict import forecast

    return forecast(artifact, history_days=history_days)
