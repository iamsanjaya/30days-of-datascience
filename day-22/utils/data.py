"""
utils/data.py — Cats vs Dogs data loading & stratified subsetting.

Used by 01_create_subsets.py and 02_train_subset_models.py / 03_augmentation_experiment.py.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    """Load a Day 21 split manifest (filepath,label columns, absolute paths)."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {csv_path}")
    return df


def stratified_subset(
    df: pd.DataFrame, fraction: float, seed: int = config.SEED
) -> pd.DataFrame:
    """Return a class-balanced subset of df at the given fraction (0 < fraction <= 1)."""
    if fraction >= 1.0:
        return df.reset_index(drop=True)

    subset_df, _ = train_test_split(
        df,
        train_size=fraction,
        stratify=df["label"],
        random_state=seed,
    )
    return cast(pd.DataFrame, subset_df).reset_index(drop=True)


def _load_and_preprocess(
    filepath: tf.Tensor, label: tf.Tensor, img_size: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, label


def _build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            # tf.keras.layers.RandomRotation(config.AUG_ROTATION),
            # tf.keras.layers.RandomZoom(config.AUG_ZOOM),
            # tf.keras.layers.RandomContrast(config.AUG_CONTRAST),
        ],
        name="augmentation",
    )


def make_dataset(
    df: pd.DataFrame,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = True,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (image, label) from a filepath/label DataFrame."""
    filepaths = df["filepath"].to_numpy()
    labels = df["label"].to_numpy().astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed)

    ds = ds.map(
        lambda fp, lbl: _load_and_preprocess(fp, lbl, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        aug_layer = _build_augmentation_layer()
        ds = ds.map(
            lambda img, lbl: (aug_layer(img, training=True), lbl),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
