"""
api.py
------
FastAPI backend that serves the trained model for real-time inference.
This simulates how the model would be deployed on a factory production
line: a camera captures an image, sends it to this API, and gets back
an instant defective/good decision with a confidence score.

SAFETY DESIGN: the final verdict combines TWO independent checks:
  1. The CNN's own good/defective prediction.
  2. An out-of-distribution (OOD) check comparing the image's basic
     visual statistics against known-good reference images (see
     src/compute_reference_stats.py and src/utils.py::compute_ood_score).
An image is only reported as "good" if BOTH checks agree it looks like
a normal, in-distribution product photo. This means a completely
unrelated image (a logo, a random photo, etc.) is correctly flagged as
defective/failed-inspection instead of being misclassified as "good",
even though the CNN alone was never trained to recognize such inputs.

Run with:
    uvicorn app.api:app --reload --port 8000

Then test at: http://127.0.0.1:8000/docs (interactive Swagger UI)
"""

import io
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src.utils import compute_ood_score

MODEL_PATH = "models/best_model.keras"
REFERENCE_STATS_PATH = "models/reference_stats.json"
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "defective", 1: "good"}  # alphabetical order used by Keras

# Fallback threshold used ONLY if reference_stats.json doesn't contain a
# calibrated one (e.g. an older stats file). The calibrated, data-driven
# threshold from compute_reference_stats.py is strongly preferred, since
# a fixed number can be too strict or too loose depending on how visually
# uniform your specific product images naturally are.
DEFAULT_OOD_THRESHOLD = 6.0

app = FastAPI(
    title="AI-Powered Visual Defect Detection API",
    description="Upload a product image and get a real-time defect prediction.",
    version="1.0.0",
)

# Allow the Streamlit app (or any frontend) to call this API from a
# different origin/port during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The model and reference stats are loaded ONCE at startup, not on every
# request, since loading a deep learning model from disk is slow
# (~1-2 seconds) and would make the API unusably slow under real-time
# inspection load.
model = None
reference_stats = None
ood_threshold = DEFAULT_OOD_THRESHOLD


class PredictionResponse(BaseModel):
    predicted_class: str
    is_defective: bool
    confidence: float
    defective_probability: float
    is_anomalous: bool
    ood_score: float
    reason: str


@app.on_event("startup")
def load_model():
    global model, reference_stats, ood_threshold

    if not os.path.exists(MODEL_PATH):
        print(
            f"WARNING: No model found at '{MODEL_PATH}'. "
            f"Run `python src/train.py` before using the API."
        )
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")

    if not os.path.exists(REFERENCE_STATS_PATH):
        print(
            f"WARNING: No reference stats found at '{REFERENCE_STATS_PATH}'. "
            f"Run `python src/compute_reference_stats.py` to enable the "
            f"out-of-distribution safety check. Predictions will rely on "
            f"the CNN alone until then."
        )
    else:
        with open(REFERENCE_STATS_PATH) as f:
            reference_stats = json.load(f)
        ood_threshold = reference_stats.get("ood_threshold", DEFAULT_OOD_THRESHOLD)
        print(
            f"Reference statistics loaded. OOD safety check is active "
            f"(calibrated threshold: {ood_threshold:.2f})."
        )

    print("API is ready for inference.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Converts raw uploaded bytes into a model-ready array.

    NOTE: we intentionally keep raw 0-255 pixel values here (no /255
    division). The model itself applies mobilenet_v2.preprocess_input
    internally, which expects that raw range and does its own scaling
    to [-1, 1]. Dividing by 255 here as well would double-normalize the
    image and break predictions.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.array(image).astype("float32")
    return np.expand_dims(array, axis=0)  # add batch dimension -> (1, H, W, 3)


@app.get("/")
def root():
    return {
        "message": "AI-Powered Visual Defect Detection API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "ood_check_active": reference_stats is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns a defect prediction.

    The final `is_defective` verdict is TRUE if EITHER:
      - the CNN predicts "defective", OR
      - the image is flagged as out-of-distribution (doesn't statistically
        resemble a real product photo), meaning it may not even be a valid
        input for this inspection system.

    Example (curl):
        curl -X POST "http://127.0.0.1:8000/predict" \\
             -F "file=@sample_image.png"
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first with `python src/train.py`.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # ---- Check 1: the CNN's own prediction ----
    # Sigmoid output: probability that the image is "good" (class 1),
    # since Keras assigns class indices alphabetically:
    # 0 = defective, 1 = good.
    prob_good = float(model.predict(input_array, verbose=0)[0][0])
    prob_defective = 1.0 - prob_good
    model_says_defective = prob_defective > 0.5

    # ---- Check 2: does this even look like a valid product photo? ----
    is_anomalous = False
    ood_score = 0.0
    if reference_stats is not None:
        ood_score = compute_ood_score(pil_image, reference_stats)
        is_anomalous = ood_score > ood_threshold

    # ---- Combine both checks: flag as defective if EITHER fires ----
    is_defective = model_says_defective or is_anomalous

    if is_anomalous and not model_says_defective:
        reason = "anomaly_detected"  # CNN said "good" but image doesn't match known product appearance
    elif model_says_defective and is_anomalous:
        reason = "model_and_anomaly"
    elif model_says_defective:
        reason = "model_detected_defect"
    else:
        reason = "passed"

    predicted_class = "defective" if is_defective else "good"
    confidence = prob_defective if model_says_defective else prob_good

    return PredictionResponse(
        predicted_class=predicted_class,
        is_defective=is_defective,
        confidence=round(confidence, 4),
        defective_probability=round(prob_defective, 4),
        is_anomalous=is_anomalous,
        ood_score=round(ood_score, 3),
        reason=reason,
    )
