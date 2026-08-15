"""
utils.py
--------
Shared helper functions:
  - plot_training_history(): visualizes accuracy/loss curves after training.
  - make_gradcam_heatmap() / overlay_gradcam(): Grad-CAM explainability,
    which highlights WHICH pixels in an image made the model think
    "defective". This is important in industrial settings — inspectors
    need to trust and verify the model's decisions, not just get a
    black-box yes/no answer.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image


def plot_training_history(history, save_path="models/training_history.png"):
    """
    Plots accuracy, loss, precision, and recall curves over epochs and
    saves the figure to disk.

    Args:
        history: the History object returned by model.fit().
        save_path (str): where to save the resulting PNG.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss", "val_loss", "Loss"),
        ("precision", "val_precision", "Precision"),
        ("recall", "val_recall", "Recall"),
    ]

    for ax, (train_key, val_key, title) in zip(axes.flat, metrics):
        if train_key in history.history:
            ax.plot(history.history[train_key], label="train")
        if val_key in history.history:
            ax.plot(history.history[val_key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    """
    Computes a Grad-CAM heatmap showing which regions of the image most
    influenced the model's prediction.

    Args:
        img_array (np.ndarray): RAW (un-normalized, 0-255 range) image
            batch, shape (1, H, W, 3). Preprocessing to [-1, 1] happens
            inside this function, matching what the model does internally.
        model (tf.keras.Model): the trained model.
        last_conv_layer_name (str): name of the last convolutional layer
            in the base network (MobileNetV2's is "Conv_1" by default).

    Returns:
        np.ndarray: 2D heatmap, values in [0, 1].

    Implementation note: MobileNetV2 is nested INSIDE our model as a
    single sub-model "layer". Because of that, `base_model.get_layer(x).output`
    points to tensors from the base model's own standalone graph (built
    when MobileNetV2 was first constructed) — NOT the graph created when
    it was called inside our outer model. Using those tensors directly
    raises "not connected to inputs". The fix: build a small model using
    the base model's OWN input/output pair (which IS internally
    consistent), then manually replay the remaining "head" layers
    (GlobalAveragePooling2D, Dense, etc.) inside the same GradientTape so
    gradients can flow from the final prediction back to the conv layer.
    """
    # Find the base MobileNetV2 sub-model (it's nested inside our model)
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 base layer in model.")

    # A model that outputs BOTH the target conv layer's activations AND
    # the base model's own final output, using base_model's own internal
    # (self-consistent) input/output graph.
    conv_layer = base_model.get_layer(last_conv_layer_name)
    activation_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output],
    )

    # Replicate the same preprocessing the full model applies internally
    # (mobilenet_v2.preprocess_input expects raw 0-255 pixel values).
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_array, tf.float32)
    )

    # Collect the "head" layers that come AFTER the base model in the
    # full model (GlobalAveragePooling2D, Dropout, Dense, BatchNorm, ...),
    # so we can manually forward-pass through them to reach the final
    # prediction, keeping everything inside one GradientTape.
    head_layers = []
    found_base = False
    for layer in model.layers:
        if layer is base_model:
            found_base = True
            continue
        if found_base:
            head_layers.append(layer)

    with tf.GradientTape() as tape:
        conv_output, base_output = activation_model(preprocessed)
        tape.watch(conv_output)

        x = base_output
        for layer in head_layers:
            # Run BatchNorm/Dropout in inference mode, matching how
            # predictions are made outside of training.
            if isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.Dropout)):
                x = layer(x, training=False)
            else:
                x = layer(x)

        loss = x[:, 0]  # sigmoid output (probability of the "good" class)

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on top of the original image for
    human-readable visualization.

    Args:
        original_img (np.ndarray): original RGB image, shape (H, W, 3), uint8.
        heatmap (np.ndarray): 2D heatmap from make_gradcam_heatmap().
        alpha (float): blending strength of the heatmap overlay.

    Returns:
        np.ndarray: RGB image (uint8) with heatmap overlay.
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = np.uint8(original_img * (1 - alpha) + heatmap_colored * alpha)
    return overlaid


def extract_image_features(image: Image.Image) -> np.ndarray:
    """
    Extracts a small, fast feature vector describing an image's overall
    visual appearance: average color, color variation, and edge density
    (how "busy"/textured the image is).

    This is used two ways:
      1. To build a "reference fingerprint" of what a real, in-distribution
         product image looks like (see compute_reference_stats.py).
      2. To score a NEW image against that fingerprint at inference time,
         so we can catch inputs that aren't even the right kind of image
         (wrong object, wrong background, a logo, a random photo, etc.)
         BEFORE trusting the CNN's good/defective prediction.

    Returns:
        np.ndarray of shape (7,):
        [mean_r, mean_g, mean_b, std_r, std_g, std_b, edge_density]
    """
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype("float32")

    mean_rgb = arr.mean(axis=(0, 1))   # overall average color
    std_rgb = arr.std(axis=(0, 1))     # how much color varies across the image

    gray = cv2.cvtColor(arr.astype("uint8"), cv2.COLOR_RGB2GRAY)
    # Laplacian variance is a classic, cheap measure of edge/texture density —
    # sharp edges and fine detail produce a high value; flat, smooth regions
    # produce a low value.
    edge_density = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.concatenate([mean_rgb, std_rgb, [edge_density]])


def compute_ood_score(image: Image.Image, reference_stats: dict) -> float:
    """
    Computes how many "standard deviations away" an image's appearance is
    from the typical appearance of known-good training images.

    A HIGH score means: this image doesn't statistically resemble anything
    the model was actually trained on — it might be the wrong object, a
    logo, a random photo, or a completely different scene. In that case we
    should NOT trust the CNN's raw "good" prediction, since the model was
    never taught what to do with inputs like this — it was forced to guess.

    Args:
        image (PIL.Image): the uploaded image, any size/mode.
        reference_stats (dict): loaded from models/reference_stats.json,
            containing the mean and standard deviation of each feature
            across all known 'good' training images.

    Returns:
        float: a distance score. 0 = looks exactly like an average 'good'
            training image. Higher = increasingly unlike anything the
            model has seen. In practice, real product photos typically
            score under ~2-3; unrelated images (logos, random photos)
            often score far higher.
    """
    feature = extract_image_features(image)
    ref_mean = np.array(reference_stats["mean"])
    ref_std = np.array(reference_stats["std"])

    z_scores = (feature - ref_mean) / ref_std
    distance = float(np.sqrt(np.sum(z_scores ** 2)))
    return distance
