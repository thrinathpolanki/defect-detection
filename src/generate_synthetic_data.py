"""
generate_synthetic_data.py
---------------------------
Generates a small synthetic image dataset so the ENTIRE pipeline
(preprocessing -> training -> evaluation -> deployment) can be run
end-to-end immediately, without waiting to source and label a real
factory dataset first.

This mimics a simple manufactured metal part (a circle on a plain
background). "Good" images are clean circles. "Defective" images have
the same circle but with a random scratch, dent (dark blob), or crack
drawn on top — simulating common visual manufacturing defects.

IMPORTANT FOR YOUR PROJECT SUBMISSION:
Once you're comfortable the pipeline works, replace the contents of
data/train/* and data/test/* with a REAL labeled dataset, e.g.:
  - MVTec AD (Anomaly Detection) dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
  - Kaggle "Casting Product Image Data for Quality Inspection"
  - Or your own photographed products (see README for labeling tips)
The rest of the code (model, training, evaluation, app) needs ZERO
changes — it only expects the good/defective folder structure.
"""

import os
import random
from PIL import Image, ImageDraw

# Reproducibility
random.seed(42)

IMG_SIZE = 300
BG_COLOR = (235, 235, 235)   # light gray background (like a metal plate)
PART_COLOR = (90, 90, 200)   # the "product" itself (a colored circle)


def _draw_base_part(draw):
    """Draws the base 'product' shape: a circle roughly centered."""
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = random.randint(90, 110)
    # Slight random jitter in position, simulating imperfect camera framing
    cx += random.randint(-10, 10)
    cy += random.randint(-10, 10)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PART_COLOR, outline=(50, 50, 120), width=3)
    return cx, cy, r


def _add_scratch(draw, cx, cy, r):
    """Simulates a scratch: a thin light-colored jagged line."""
    x1 = cx + random.randint(-r, r)
    y1 = cy + random.randint(-r, r)
    x2 = x1 + random.randint(-60, 60)
    y2 = y1 + random.randint(-60, 60)
    draw.line([x1, y1, x2, y2], fill=(255, 255, 255), width=random.randint(2, 4))


def _add_dent(draw, cx, cy, r):
    """Simulates a dent: a small dark irregular blob."""
    x = cx + random.randint(-r // 2, r // 2)
    y = cy + random.randint(-r // 2, r // 2)
    size = random.randint(10, 22)
    draw.ellipse([x - size, y - size, x + size, y + size], fill=(20, 20, 20))


def _add_crack(draw, cx, cy, r):
    """Simulates a crack: a jagged multi-segment dark line."""
    x, y = cx, cy
    color = (10, 10, 10)
    for _ in range(5):
        nx = x + random.randint(-25, 25)
        ny = y + random.randint(-25, 25)
        draw.line([x, y, nx, ny], fill=color, width=2)
        x, y = nx, ny


def generate_image(defective: bool) -> Image.Image:
    """Generates a single synthetic product image."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    cx, cy, r = _draw_base_part(draw)

    if defective:
        # Randomly apply 1-2 types of defects for visual variety
        defect_fns = random.sample([_add_scratch, _add_dent, _add_crack], k=random.choice([1, 2]))
        for fn in defect_fns:
            fn(draw, cx, cy, r)

    return img


def generate_dataset(root="data", n_train_per_class=200, n_test_per_class=50):
    """
    Populates data/train/{good,defective} and data/test/{good,defective}
    with synthetically generated images.
    """
    splits = {
        "train": n_train_per_class,
        "test": n_test_per_class,
    }

    for split, count in splits.items():
        for label, is_defective in [("good", False), ("defective", True)]:
            out_dir = os.path.join(root, split, label)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(count):
                img = generate_image(defective=is_defective)
                img.save(os.path.join(out_dir, f"{label}_{i:04d}.png"))
            print(f"Generated {count} images -> {out_dir}")

    print("\nSynthetic dataset generation complete.")
    print(f"Total images: {(n_train_per_class + n_test_per_class) * 2}")


if __name__ == "__main__":
    generate_dataset()
