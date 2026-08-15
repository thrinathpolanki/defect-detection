"""
streamlit_app.py
-----------------
A user-friendly web UI for demoing the defect detection model — ideal
for showing your mentors/internship reviewers a live, interactive demo.

SAFETY DESIGN: the final verdict combines TWO independent checks:
  1. The CNN's own good/defective prediction.
  2. An out-of-distribution (OOD) check comparing the image's basic
     visual statistics against known-good reference images.
An image is only shown as "good" if BOTH checks agree. A completely
unrelated image (a logo, a random photo) is correctly flagged as
defective/failed-inspection, since the CNN alone has no way to say
"I don't recognize this at all."

Run with:
    streamlit run app/streamlit_app.py --server.enableXsrfProtection false

Features:
  - Upload a product image and get an instant prediction.
  - Visual confidence gauge.
  - Grad-CAM heatmap showing WHERE the model "looked" to make its
    decision (explainability — builds trust in the system).
  - Anomaly/out-of-distribution flag for inputs that don't resemble a
    real product photo at all.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.utils import make_gradcam_heatmap, overlay_gradcam, compute_ood_score

MODEL_PATH = "models/best_model.keras"
REFERENCE_STATS_PATH = "models/reference_stats.json"
IMG_SIZE = (224, 224)

# Fallback threshold used ONLY if reference_stats.json doesn't contain a
# calibrated one (e.g. an older stats file). The calibrated, data-driven
# threshold from compute_reference_stats.py is strongly preferred.
DEFAULT_OOD_THRESHOLD = 6.0

st.set_page_config(
    page_title="AI Visual Defect Detection",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Loads the trained model once and caches it across reruns."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_reference_stats():
    """Loads the known-good image fingerprint once and caches it."""
    if not os.path.exists(REFERENCE_STATS_PATH):
        return None
    with open(REFERENCE_STATS_PATH) as f:
        return json.load(f)


def preprocess(image: Image.Image) -> np.ndarray:
    """
    NOTE: no /255 division here — the model applies mobilenet_v2's own
    preprocessing internally (expects raw 0-255 values). Dividing here
    too would double-normalize the image and break predictions.
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image).astype("float32")
    return np.expand_dims(array, axis=0)


def main():
    st.title("🔍 AI-Powered Visual Defect Detection")
    st.caption(
        "Upload an image of a manufactured product to check whether it passes "
        "quality inspection — powered by a MobileNetV2 CNN with transfer learning, "
        "plus an anomaly-detection safety check."
    )

    model = load_model()
    reference_stats = load_reference_stats()
    ood_threshold = (
        reference_stats.get("ood_threshold", DEFAULT_OOD_THRESHOLD)
        if reference_stats is not None
        else DEFAULT_OOD_THRESHOLD
    )

    with st.sidebar:
        st.header("ℹ️ About this system")
        st.write(
            "This model was trained to distinguish **good** products from "
            "**defective** ones (scratches, dents, cracks) using computer vision."
        )
        st.write("**Architecture:** MobileNetV2 (transfer learning)")
        st.write("**Input size:** 224 × 224 pixels")
        st.write(
            "**Safety check:** any image that doesn't statistically resemble "
            "a real product photo (wrong object, logo, unrelated image) is "
            "automatically flagged as defective, even if the CNN alone would "
            "have guessed 'good'."
        )

        if model is None:
            st.error(
                "⚠️ No trained model found.\n\n"
                "Run these commands first:\n"
                "```\npython src/generate_synthetic_data.py\npython src/train.py\n```"
            )
        else:
            st.success("✅ Model loaded and ready.")

        if reference_stats is None:
            st.warning(
                "⚠️ No reference stats found — anomaly detection is OFF.\n\n"
                "Run:\n```\npython src/compute_reference_stats.py\n```"
            )
        else:
            st.success("✅ Anomaly detection active.")

    if model is None:
        st.warning(
            "No trained model available yet. Please train the model first "
            "(see instructions in the sidebar), then refresh this page."
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a product image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_np = np.array(image.convert("RGB").resize(IMG_SIZE))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        # ---------------- Check 1: CNN prediction ----------------
        input_array = preprocess(image)
        with st.spinner("Analyzing image..."):
            prob_good = float(model.predict(input_array, verbose=0)[0][0])
        prob_defective = 1.0 - prob_good
        model_says_defective = prob_defective > 0.5

        # ---------------- Check 2: out-of-distribution check ----------------
        is_anomalous = False
        ood_score = 0.0
        if reference_stats is not None:
            ood_score = compute_ood_score(image, reference_stats)
            is_anomalous = ood_score > ood_threshold

        # ---------------- Combine both checks ----------------
        is_defective = model_says_defective or is_anomalous

        with col2:
            st.subheader("Prediction")
            if is_defective:
                st.error("### ❌ DEFECTIVE")
                if is_anomalous and not model_says_defective:
                    st.caption(
                        "⚠️ Flagged by the anomaly check: this image doesn't "
                        "statistically resemble the trained product's appearance "
                        "(e.g. wrong object, unrelated image)."
                    )
                confidence = prob_defective if model_says_defective else 1.0
                st.metric("Confidence", f"{confidence * 100:.1f}%")
            else:
                st.success("### ✅ GOOD")
                st.metric("Confidence", f"{prob_good * 100:.1f}%")

            st.progress(
                min(prob_defective, 1.0),
                text=f"Model defect probability: {prob_defective * 100:.1f}%",
            )

            if reference_stats is not None:
                st.caption(f"Anomaly score: {ood_score:.2f}  (flag threshold: {ood_threshold:.2f})")

        # ---------------- Grad-CAM explainability ----------------
        with col3:
            st.subheader("Model Focus (Grad-CAM)")
            try:
                heatmap = make_gradcam_heatmap(input_array, model)
                overlay = overlay_gradcam(original_np, heatmap)
                st.image(overlay, use_container_width=True, caption="Red/yellow = high influence on decision")
            except Exception as e:
                st.info(f"Grad-CAM visualization unavailable: {e}")

    st.divider()
    st.caption(
        "Built as a full-stack AI/ML demo: image preprocessing → CNN training → "
        "evaluation (precision/recall) → anomaly detection → real-time deployment "
        "(FastAPI + Streamlit)."
    )


if __name__ == "__main__":
    main()
