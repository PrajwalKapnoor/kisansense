"""Disease inference: image in, ranked classes + confidence + Grad-CAM out."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from kisansense.common.config import ARTIFACTS_DIR, IMAGE_SIZE
from kisansense.disease.gradcam import compute_gradcam, overlay

MODEL_PATH = ARTIFACTS_DIR / "disease_model.keras"
CLASSES_PATH = ARTIFACTS_DIR / "disease_classes.json"

# Below this top-class probability the app says "uncertain" instead of naming a disease.
# Model confidence is not accuracy: this is a guard rail, not a guarantee.
UNCERTAIN_BELOW = 0.60

INFO = {
    "Tomato___Bacterial_spot": ("Bacterial spot", "Small dark, water-soaked spots on leaves that can merge and turn brown. Spreads in warm, wet weather."),
    "Tomato___Early_blight": ("Early blight", "Brown spots with concentric rings (a target look), usually on older leaves first."),
    "Tomato___Late_blight": ("Late blight", "Large, irregular, dark water-soaked patches that spread quickly in cool, humid weather."),
    "Tomato___Leaf_Mold": ("Leaf mold", "Pale yellow patches on the upper leaf surface with olive-grey mold underneath. Favoured by high humidity."),
    "Tomato___Septoria_leaf_spot": ("Septoria leaf spot", "Many small round spots with dark borders and lighter centres, mostly on lower leaves."),
    "Tomato___healthy": ("Healthy", "No disease pattern recognised in this image."),
}
REMINDER = (
    "This is an experimental image classifier trained on lab photos. It can be wrong on field images. "
    "Confirm with a local agricultural extension officer before acting."
)


@lru_cache(maxsize=1)
def load(model_path: str = str(MODEL_PATH), classes_path: str = str(CLASSES_PATH)):
    from tensorflow import keras

    model = keras.models.load_model(model_path)
    classes = json.loads(Path(classes_path).read_text())
    return model, classes


def prepare_image(img: Image.Image, size=IMAGE_SIZE) -> np.ndarray:
    """RGB, resized, float32 in [0, 255]."""
    return np.asarray(img.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)


def predict_image(img: Image.Image, artifact=None, with_cam: bool = True, top_k: int = 3) -> dict:
    model, classes = artifact or load()
    arr = prepare_image(img, tuple(model.input_shape[1:3]))
    probs = model.predict(arr[None], verbose=0)[0]
    order = probs.argsort()[::-1][:top_k]
    top = [{"class": classes[i], "name": INFO.get(classes[i], (classes[i], ""))[0], "probability": float(probs[i])} for i in order]
    best = classes[int(order[0])]
    confidence = float(probs[order[0]])
    uncertain = confidence < UNCERTAIN_BELOW
    out = {
        "prediction": None if uncertain else best,
        "name": "Uncertain" if uncertain else INFO.get(best, (best, ""))[0],
        "description": "" if uncertain else INFO.get(best, ("", ""))[1],
        "confidence": confidence,
        "uncertain": uncertain,
        "top": top,
        "reminder": REMINDER,
    }
    if with_cam and hasattr(model, "get_layer"):
        try:
            cam, _, _ = compute_gradcam(model, arr, int(order[0]))
            out["gradcam"] = overlay(arr.astype(np.uint8), cam)
        except ValueError:  # model without a backbone/head split (e.g. the baseline CNN)
            out["gradcam"] = None
    return out
