import base64
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from backend.app.deps import disease_artifact
from backend.app.schemas import DiseaseResponse

router = APIRouter()

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@router.post("/predict", response_model=DiseaseResponse)
def predict_disease(file: UploadFile = File(...), artifact=Depends(disease_artifact)):
    from kisansense.disease.predict import predict_image

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force a full decode so truncated or fake files fail here
    except Exception as e:
        raise HTTPException(status_code=400, detail="The file is not a readable image.") from e

    result = predict_image(img, artifact=artifact)
    cam = result.pop("gradcam", None)
    result["gradcam_png_base64"] = None
    if cam is not None:
        buf = io.BytesIO()
        Image.fromarray(cam).save(buf, format="PNG")
        result["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
    return result
