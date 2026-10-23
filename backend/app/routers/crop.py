from fastapi import APIRouter, Depends, HTTPException

from backend.app.deps import crop_artifact
from backend.app.schemas import CropRequest, CropResponse

router = APIRouter()

NOTE = (
    "Scores are model probabilities learned from a clean public dataset. They do not guarantee a crop "
    "will do well on your land. Check with a local agronomist or soil-testing lab."
)


@router.post("/recommend", response_model=CropResponse)
def recommend_crop(req: CropRequest, artifact=Depends(crop_artifact)):
    from kisansense.crop.predict import InvalidInputError, recommend

    try:
        recs = recommend(req.model_dump(), top_k=3, artifact=artifact)
    except InvalidInputError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return CropResponse(recommendations=recs, note=NOTE)
