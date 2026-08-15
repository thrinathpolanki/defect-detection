"""
compute_reference_stats.py
----------------------------
Builds a lightweight statistical "fingerprint" of what a NORMAL,
in-distribution product image looks like, using only the 'good' training
images.

WHY THIS EXISTS:
A CNN classifier only ever has two buckets to choose between: good or
defective. If you show it something it has never seen before — a logo,
a random photo, a totally different object — it has NO way to say "I
don't recognize this." It is forced to place the input into whichever
bucket the pixels happen to statistically lean toward, often with a
misleadingly confident score.

To make the system safer for real inspection use, we add a SECOND,
independent check that runs alongside the CNN: this script computes
simple image statistics (average color, color variation, edge/texture
density) across every known-good training image, and saves the mean and
standard deviation of each statistic to models/reference_stats.json.

At inference time (see app/api.py and app/streamlit_app.py), any new
image is compared against this fingerprint. If it deviates too far from
what a real product image should look like, the system flags it as
DEFECTIVE — "does not match expected product appearance" — rather than
risk quietly passing an unrecognized, out-of-distribution image as GOOD.

This mirrors real quality-control philosophy: "when in doubt, flag it
for human review" is far safer than "when in doubt, let it pass."

Run with (after you have training images in data/train/good/):
    python src/compute_reference_stats.py
"""

import os
import sys
import json
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from src.utils import extract_image_features

GOOD_DIR = "data/train/good"
OUTPUT_PATH = "models/reference_stats.json"


def main():
    paths = glob.glob(os.path.join(GOOD_DIR, "*"))
    if not paths:
        raise FileNotFoundError(
            f"No images found in '{GOOD_DIR}'. Generate or add your dataset "
            f"first (e.g. `python src/generate_synthetic_data.py`), then re-run this script."
        )

    print(f"Computing reference statistics from {len(paths)} 'good' images...")
    features = []
    for path in paths:
        try:
            img = Image.open(path)
            features.append(extract_image_features(img))
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    features = np.stack(features)  # shape (N, 7)
    mean = features.mean(axis=0)
    # Add a small epsilon so we never divide by zero later if a feature
    # happens to be perfectly constant across every training image.
    std = features.std(axis=0) + 1e-6

    # --- Calibrate the anomaly threshold from the data itself ---
    # A fixed, guessed threshold doesn't work across datasets: some
    # products (like a uniform-colored synthetic part) naturally have
    # very LITTLE variation between good images, which makes any generic
    # distance threshold too strict and flags real good images as
    # anomalies. Instead, we score every known-good training image
    # against its own fingerprint, and set the threshold safely above
    # the highest score any genuine good image received. This makes the
    # anomaly check adapt automatically to how visually consistent (or
    # varied) YOUR specific product images actually are.
    self_scores = []
    for f in features:
        z = (f - mean) / std
        self_scores.append(float(np.sqrt(np.sum(z ** 2))))
    self_scores = np.array(self_scores)

    # 30% safety margin above the worst-case genuine good image, with a
    # sensible floor so a very small/uniform dataset doesn't produce an
    # unreasonably tight threshold.
    ood_threshold = max(float(self_scores.max()) * 1.3, 5.0)

    print(f"\nSelf-check: scored all {len(paths)} 'good' training images against "
          f"their own fingerprint.")
    print(f"  Score range: {self_scores.min():.2f} - {self_scores.max():.2f} "
          f"(mean {self_scores.mean():.2f})")
    print(f"  Calibrated anomaly threshold: {ood_threshold:.2f} "
          f"(30% margin above the highest genuine 'good' score)")

    stats = {
        "feature_names": [
            "mean_r", "mean_g", "mean_b",
            "std_r", "std_g", "std_b",
            "edge_density",
        ],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_reference_images": len(paths),
        "ood_threshold": ood_threshold,
    }

    os.makedirs("models", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Reference statistics saved to {OUTPUT_PATH}")
    print(
        "The API and Streamlit app will now automatically flag images that "
        "don't statistically resemble a real product photo as 'defective'."
    )


if __name__ == "__main__":
    main()
