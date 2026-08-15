"""
train.py
--------
Main training script. Run this to train the defect detection model
end-to-end:

    python src/train.py

TWO-PHASE TRAINING:
This script trains in two phases, which matters a lot for small
datasets like this one:

  Phase 1 (warm-up): the pretrained MobileNetV2 base is completely
  frozen. Only the new classification head trains, so it can learn to
  make sense of the pretrained features WITHOUT sending noisy gradients
  back through (and scrambling) those features while it's still
  randomly initialized.

  Phase 2 (fine-tuning): the later base layers are unfrozen and trained
  further with a much smaller learning rate, gently adapting the
  pretrained features to this specific product rather than overwriting
  them.

Skipping straight to fine-tuning from epoch 1 (unfreezing early) is a
common transfer-learning mistake that causes exactly the symptom you'd
see as "trains fine, but predicts almost everything as one class on
anything it wasn't directly trained on" — the model looks like it's
learning (training accuracy climbs) while actually failing to
generalize, because the pretrained features got damaged early on.

What this script does:
  1. Loads and augments the training/validation data.
  2. PHASE 1: builds the model with a frozen base, trains the head.
  3. PHASE 2: unfreezes the later base layers, fine-tunes with a low LR.
  4. Saves the best model (by validation loss, across both phases) to
     models/best_model.keras.
  5. Saves a combined training curves plot to models/training_history.png.
"""

import os
import sys

# Allow running this script directly (python src/train.py) by adding
# the project root to the path, so imports work either way.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.data_preprocessing import get_data_generators
from src.model import build_model, unfreeze_for_finetuning
from src.utils import plot_training_history


# ---------------------- Configuration ----------------------
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
WARMUP_EPOCHS = 10       # Phase 1: train only the head
FINE_TUNE_EPOCHS = 20    # Phase 2: fine-tune unfrozen base layers
FINE_TUNE_AT = 100       # unfreeze base layers from this index onward
FINE_TUNE_LR = 1e-5      # small learning rate to avoid destroying pretrained features
MODEL_OUT_PATH = "models/best_model.keras"
# -------------------------------------------------------------


class CombinedHistory:
    """Small helper to merge Phase 1 and Phase 2 histories for a single
    combined training curves plot."""

    def __init__(self):
        self.history = {}

    def extend(self, keras_history):
        for key, values in keras_history.history.items():
            self.history.setdefault(key, []).extend(values)


def main():
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("STEP 1/4: Loading and augmenting data")
    print("=" * 60)
    train_gen, val_gen, test_gen = get_data_generators(
        TRAIN_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    print(f"Training samples:   {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples:       {test_gen.samples}")

    combined_history = CombinedHistory()

    # =====================================================================
    # PHASE 1: WARM-UP — train only the head, base model fully frozen
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 2/4: PHASE 1 - Warm-up (training only the classification head)")
    print("=" * 60)
    model = build_model(img_size=IMG_SIZE, num_classes=2)
    model.summary()

    warmup_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
        ),
    ]

    warmup_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=WARMUP_EPOCHS,
        callbacks=warmup_callbacks,
    )
    combined_history.extend(warmup_history)

    # =====================================================================
    # PHASE 2: FINE-TUNING — unfreeze later base layers, tiny learning rate
    # =====================================================================
    print("\n" + "=" * 60)
    print("STEP 3/4: PHASE 2 - Fine-tuning (unfreezing later base layers)")
    print("=" * 60)
    model = unfreeze_for_finetuning(model, fine_tune_at=FINE_TUNE_AT, learning_rate=FINE_TUNE_LR)
    model.summary()

    finetune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        # Save the best model across the ENTIRE run (both phases end up
        # comparable on val_loss, so this correctly keeps the best one).
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1
        ),
    ]

    finetune_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=finetune_callbacks,
    )
    combined_history.extend(finetune_history)

    # It's possible ModelCheckpoint never saved (e.g. if Phase 1's
    # restored-best-weights model was already excellent and Phase 2 never
    # beat it before EarlyStopping fired on epoch 1). Save explicitly as
    # a fallback so a model file always exists after this script runs.
    if not os.path.exists(MODEL_OUT_PATH):
        model.save(MODEL_OUT_PATH)

    print("\n" + "=" * 60)
    print("STEP 4/4: Saving results")
    print("=" * 60)
    plot_training_history(combined_history)
    print(f"Best model saved to: {MODEL_OUT_PATH}")
    print("\nNext steps:")
    print("  python src/compute_reference_stats.py   (rebuild anomaly-detection fingerprint)")
    print("  python src/evaluate.py                  (measure test performance)")


if __name__ == "__main__":
    main()
