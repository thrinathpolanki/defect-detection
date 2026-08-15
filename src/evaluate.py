"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set and reports the
metrics that matter most for a defect detection system:

  - Precision: of all images the model FLAGGED as defective, how many
    actually were? (Low precision = too many false alarms, wasting
    inspectors' time on parts that are actually fine.)
  - Recall: of all the TRULY defective images, how many did the model
    catch? (Low recall = defective products slip through to customers —
    usually the more costly mistake in manufacturing QA.)
  - F1-score: harmonic mean of precision and recall.
  - Confusion matrix: full breakdown of correct/incorrect predictions.
  - ROC-AUC: overall ability to distinguish defective from good, across
    all possible decision thresholds.

Run with:
    python src/evaluate.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.data_preprocessing import get_data_generators

MODEL_PATH = "models/best_model.keras"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'. Run `python src/train.py` first."
        )

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test data...")
    _, _, test_gen = get_data_generators(TRAIN_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=32)

    print("Running predictions on test set...")
    y_true = test_gen.classes  # ground-truth labels, 0/1
    y_pred_probs = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    class_names = list(test_gen.class_indices.keys())  # e.g. ['defective', 'good']

    # ---------------- Classification report ----------------
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    with open("models/evaluation_report.txt", "w") as f:
        f.write("DEFECT DETECTION MODEL - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    # ---------------- ROC-AUC ----------------
    auc = roc_auc_score(y_true, y_pred_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

    # ---------------- Confusion matrix plot ----------------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to models/confusion_matrix.png")

    # ---------------- ROC curve plot ----------------
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=150)
    plt.close()
    print("ROC curve saved to models/roc_curve.png")

    print("\nEvaluation complete. See the models/ folder for saved reports and plots.")


if __name__ == "__main__":
    evaluate()
