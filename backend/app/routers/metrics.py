"""Serves the saved evaluation results so the app can show real, measured numbers."""
import json

from fastapi import APIRouter, HTTPException

from kisansense.common.config import DOCS_DIR

router = APIRouter()
MODULES = {"crop", "price", "disease"}


@router.get("/{module}")
def get_metrics(module: str):
    if module not in MODULES:
        raise HTTPException(status_code=404, detail=f"Unknown module. Choose one of {sorted(MODULES)}.")
    path = DOCS_DIR / "results" / f"{module}_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No saved metrics for {module} yet.")
    return json.loads(path.read_text())
