"""
model.py
--------
Defines the CNN architecture used for visual defect detection.

We use MobileNetV2 (pretrained on ImageNet) as a feature extractor and
attach a custom classification head on top. This is called "transfer
learning" — instead of teaching a network to recognize edges, textures,
and shapes from scratch (which needs millions of images), we reuse a
network that already learned those basics, and only teach it the
new task: "does this specific product look normal or defective?"

TWO-PHASE TRAINING (important):
This module is designed to be used in two phases, and train.py drives
both of them:

  Phase 1 - Warm-up: build_model() returns a model with the ENTIRE
  pretrained base FROZEN. Only the new classification head (which starts
  with random weights) is trained. This matters because an untrained
  head sends large, noisy gradients backward through the network — if
  the pretrained base were unfrozen from the very first epoch, those
  noisy gradients would scramble the valuable pretrained features before
  the head has learned anything useful. This is especially damaging with
  a small dataset, which has little room to recover from that damage.

  Phase 2 - Fine-tuning: once the head has learned to make reasonable
  use of the frozen features (a handful of epochs), call
  unfreeze_for_finetuning() to unfreeze the LATER base layers and
  continue training with a much smaller learning rate, so the pretrained
  features are gently nudged toward this specific product rather than
  being overwritten.

Why MobileNetV2?
- It's small and fast enough to run in real time on a factory floor
  (even on a CPU or edge device), unlike bulkier models like ResNet50
  or Vision Transformers.
- It still reaches strong accuracy for industrial inspection tasks.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_model(img_size=(224, 224), num_classes=2):
    """
    Builds and compiles a MobileNetV2-based defect classifier for PHASE 1
    (warm-up): the entire pretrained base is frozen, and only the new
    classification head is trainable.

    Args:
        img_size (tuple): (height, width) of input images fed to the model.
        num_classes (int): 2 for binary (good vs defective), >2 for
            multi-class (e.g. scratch / dent / crack / good).

    Returns:
        tf.keras.Model: a compiled, ready-to-train Keras model with the
            base model frozen.
    """
    input_shape = img_size + (3,)

    # 1. Load MobileNetV2 pretrained on ImageNet, WITHOUT its final
    #    classification layer (include_top=False), since we want our own.
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # 2. PHASE 1: freeze the entire base model. We only train the new
    #    head on top, so we don't disturb the pretrained features with
    #    noisy gradients from a randomly-initialized head.
    base_model.trainable = False

    # 3. Build the full model: input -> preprocessing -> base -> head
    inputs = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects pixel values scaled to [-1, 1]; this layer
    # does that automatically so users can just feed in raw 0-255 images.
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # GlobalAveragePooling condenses each feature map into a single
    # number, turning a 3D feature map into a 1D feature vector.
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    if num_classes == 2:
        # Binary classification: single neuron, sigmoid -> probability
        # of "good" class (Keras assigns class 1 = good alphabetically).
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
    else:
        # Multi-class: one neuron per class, softmax -> probability
        # distribution over all defect types.
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
        loss = "categorical_crossentropy"

    model = models.Model(inputs, outputs, name="defect_detector_mobilenetv2")

    model.compile(
        # A relatively higher learning rate is fine here since ONLY the
        # small, randomly-initialized head is being trained.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def unfreeze_for_finetuning(model, fine_tune_at=100, learning_rate=1e-5):
    """
    Prepares a warmed-up model for PHASE 2 (fine-tuning): unfreezes the
    LATER layers of the pretrained base and recompiles with a much
    smaller learning rate, so the pretrained features are gently
    adjusted to this specific product rather than being overwritten.

    Call this AFTER Phase 1 warm-up training has already run for a few
    epochs (so the head is no longer sending large, noisy gradients).

    Args:
        model (tf.keras.Model): a model previously built with build_model()
            and already warmed up.
        fine_tune_at (int): layer index (within the base model) from
            which to unfreeze. Earlier layers (generic features like
            edges/colors) stay frozen; later layers (task-specific
            features) become trainable.
        learning_rate (float): should be much smaller than Phase 1's,
            since we're now updating pretrained weights, not random ones —
            large updates here would destroy useful pretrained features.

    Returns:
        tf.keras.Model: the same model object, now with part of its base
            unfrozen and recompiled for fine-tuning.
    """
    # Locate the nested MobileNetV2 sub-model. We search by name
    # substring (its auto-generated name is something like
    # "mobilenetv2_1.00_224") rather than an exact name, since Keras
    # doesn't reliably let us rename a pretrained application model after
    # construction — this matches the same lookup approach used in
    # utils.py's Grad-CAM implementation, so both stay consistent.
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 base layer in model.")

    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile is REQUIRED after changing any layer's trainable flag,
    # for the change to take effect.
    loss = model.loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


if __name__ == "__main__":
    # Running `python src/model.py` directly prints the architecture,
    # useful as a quick sanity check.
    m = build_model()
    m.summary()
