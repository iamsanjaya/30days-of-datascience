"""
utils/niche_data.py — Devanagari digit data loading & tiny-pool sampling.

Used by 05_prepare_niche_dataset.py and all niche training/eval scripts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def scan_digit_classes(split_dir: Path) -> pd.DataFrame:
    """Scan split_dir/digit_0 .. digit_9 into a filepath/label DataFrame.

    Label is the integer digit (0-9), derived from the class folder name.
    """
    records: list[dict] = []
    for class_name in config.NICHE_CLASSES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}. "
                "Check config.NICHE_RAW_DATA_DIR / NICHE_TRAIN_SUBDIR / "
                "NICHE_TEST_SUBDIR against your actual extracted dataset."
            )
        label = int(class_name.split("_")[1])
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                records.append({"filepath": str(img_path), "label": label})

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError(f"No images found under {split_dir}")
    return df


def sample_non_overlapping_pools(
    df: pd.DataFrame,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    unlabeled_per_class: int,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """Carve four non-overlapping, class-balanced pools out of df.

    Order matters: train pool is sampled first (the deliberately tiny,
    constrained set), then val, then test, then the simulated-unlabeled pool
    — each drawn from what remains so no image appears in two pools.
    """
    rng = np.random.RandomState(seed)
    remaining = df.copy()
    pools: dict[str, pd.DataFrame] = {}

    for pool_name, per_class in [
        ("train", train_per_class),
        ("val", val_per_class),
        ("test", test_per_class),
        ("unlabeled", unlabeled_per_class),
    ]:
        sampled_parts = []
        for label in sorted(remaining["label"].unique()):
            class_rows = remaining[remaining["label"] == label]
            n = min(per_class, len(class_rows))
            sampled_parts.append(
                class_rows.sample(n=n, random_state=rng.randint(0, 1_000_000))
            )
        pool_df = pd.concat(sampled_parts).reset_index(drop=True)
        pools[pool_name] = pool_df
        remaining = remaining[
            ~remaining["filepath"].isin(pool_df["filepath"])
        ].reset_index(drop=True)

    return pools


def _load_grayscale_to_rgb(
    filepath: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=1, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.image.grayscale_to_rgb(image)
    image = image / 255.0
    return image, label


def _build_niche_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="niche_augmentation",
    )


def make_niche_dataset(
    df: pd.DataFrame,
    img_size: tuple[int, int] = config.NICHE_IMG_SIZE,
    batch_size: int = config.NICHE_BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = True,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (image, one_hot_label) for the digit classifier."""
    filepaths = df["filepath"].to_numpy()
    labels = tf.keras.utils.to_categorical(
        df["label"].to_numpy(), num_classes=config.NICHE_NUM_CLASSES
    )

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(df), 1), seed=seed)

    ds = ds.map(
        lambda fp, lbl: _load_grayscale_to_rgb(fp, lbl, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        aug_layer = _build_niche_augmentation_layer()
        ds = ds.map(
            lambda img, lbl: (aug_layer(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def predict_with_tta(
    model: tf.keras.Model,
    df: pd.DataFrame,
    img_size: tuple[int, int] = config.NICHE_IMG_SIZE,
    num_augmentations: int = config.TTA_NUM_AUGMENTATIONS,
    seed: int = config.SEED,
) -> tuple[float, np.ndarray]:
    """Test-time augmentation: average predictions over N augmented copies of
    each test image (plus the original), then take the argmax.

    Returns (accuracy, predicted_labels).
    """
    tf.random.set_seed(seed)
    aug_layer = _build_niche_augmentation_layer()
    true_labels = df["label"].to_numpy()
    predicted_labels = np.zeros(len(df), dtype=int)

    for i, filepath in enumerate(df["filepath"].to_numpy()):
        raw = tf.io.read_file(filepath)
        image = tf.io.decode_image(raw, channels=1, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.image.grayscale_to_rgb(image)
        image = image / 255.0

        variants = [image] + [
            aug_layer(tf.expand_dims(image, 0), training=True)[0]
            for _ in range(num_augmentations)
        ]
        batch = tf.stack(variants, axis=0)
        probs = model.predict(batch, verbose=0)
        averaged_probs = np.mean(probs, axis=0)
        predicted_labels[i] = int(np.argmax(averaged_probs))

    accuracy = float(np.mean(predicted_labels == true_labels))
    return accuracy, predicted_labels
