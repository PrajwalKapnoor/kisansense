"""KisanSense API.

Run from the project root:
    uvicorn backend.app.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

# Make `kisansense` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import FastAPI  # noqa: E402

from backend.app import deps  # noqa: E402
from backend.app.routers import crop, disease, metrics, price  # noqa: E402

app = FastAPI(
    title="KisanSense API",
    version="0.1.0",
    description=(
        "Experimental agricultural decision support: tomato leaf disease detection, crop recommendation "
        "and Karnataka tomato price trends. Not agricultural or financial advice."
    ),
)

app.include_router(crop.router, prefix="/api/v1/crop", tags=["crop"])
app.include_router(disease.router, prefix="/api/v1/disease", tags=["disease"])
app.include_router(price.router, prefix="/api/v1/price", tags=["price"])
app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])


@app.get("/health", tags=["system"])
def health():
    """Which model files are present (does not load them)."""
    models = {
        "crop": deps.CROP_FILE.exists(),
        "price": deps.PRICE_FILE.exists(),
        "disease": deps.DISEASE_FILE.exists() and deps.DISEASE_CLASSES_FILE.exists(),
    }
    return {"status": "ok", "models": models}
