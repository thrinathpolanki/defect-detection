"""
data_preprocessing.py
----------------------
Handles loading images from disk, preprocessing, and data augmentation.

Expected folder layout (already created for you under data/):

    data/train/good/          <- normal product images
    data/train/defective/     <- defective product images
    data/test/good/
    data/test/defective/

We use Keras' ImageDataGenerator to:
  1. Rescale pixel values from [0, 255] to [0, 1].
  2. Apply augmentation (rotation, flips, zoom, brightness shifts) ONLY
     to the training set. This artificially expands the dataset and
     teaches the model to be robust to real-world variation: a camera
     on a production line won't always capture the product at the
     exact same angle, lighting, or position.
  3. Leave validation/test data un-augmented, so the metrics we report
     reflect true, real-world performance (not an "easier" augmented
     version of the images).
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators(train_dir, test_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Creates train / validation / test data generators.

    Args:
        train_dir (str): path to training data (with class subfolders).
        test_dir (str): path to held-out test data (with class subfolders).
        img_size (tuple): target image size (height, width) fed to the CNN.
        batch_size (int): number of images per training batch.
        val_split (float): fraction of train_dir reserved for validation.

    Returns:
        tuple: (train_gen, val_gen, test_gen) Keras DirectoryIterators.
               Class indices are alphabetical, so with folders
               "defective" and "good", index 0 = defective, 1 = good
               (the generators' .class_indices attribute confirms this).
    """
    # NOTE: We deliberately do NOT set rescale=1./255 here. The model itself
    # (see model.py) applies `mobilenet_v2.preprocess_input`, which expects
    # raw 0-255 pixel values and does its own scaling to [-1, 1] internally.
    # Rescaling here AND inside the model would double-normalize every
    # image, crushing all pixel values toward the same number and making
    # the model unable to distinguish images at all.
    train_datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        validation_split=val_split,
    )

    # No augmentation for test data — we want to measure real performance.
    # Also no rescale, for the same reason as above (model normalizes internally).
    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    print(f"Class index mapping: {train_gen.class_indices}")
    return train_gen, val_gen, test_gen
