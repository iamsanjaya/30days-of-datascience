"""Day 20 — Data loading for the CIFAR-10 PNG-folder dataset.

Expects the Kaggle "CIFAR-10 PNGs in folders" layout:
    data/raw/cifar10/train/<class_name>/*.png
    data/raw/cifar10/test/<class_name>/*.png
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import config


def _load_class_folder(class_dir: Path, image_size: int) -> np.ndarray:
    """Load every PNG in a single class folder into a uint8 array."""
    paths = sorted(class_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {class_dir}")

    images = np.empty((len(paths), image_size, image_size, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            images[i] = np.array(img.convert("RGB").resize((image_size, image_size)))
    return images


def load_split(
    split_dir: Path, image_size: int = config.IMAGE_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an entire CIFAR-10 split (train/ or test/) of class subfolders."""
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folder at {split_dir}. "
            "Download the Kaggle 'CIFAR-10 PNGs in folders' dataset and unzip it "
            "there — see README.md for the exact layout."
        )

    images_by_class, labels_by_class = [], []
    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_images = _load_class_folder(split_dir / class_name, image_size)
        images_by_class.append(class_images)
        labels_by_class.append(np.full(len(class_images), label_idx, dtype=np.int64))

    return np.concatenate(images_by_class, axis=0), np.concatenate(
        labels_by_class, axis=0
    )


def disjoint_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = config.RANDOM_SEED,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split into non-overlapping train/val pools, stratified per class.

    Doing this split *before* any further subsampling guarantees the overfit
    train subset and the val subset never share an image.
    """
    rng = np.random.default_rng(seed)
    train_idx_parts, val_idx_parts = [], []

    for label_idx in range(config.NUM_CLASSES):
        class_idx = np.where(labels == label_idx)[0].copy()
        rng.shuffle(class_idx)
        split_point = int(len(class_idx) * (1 - val_fraction))
        train_idx_parts.append(class_idx[:split_point])
        val_idx_parts.append(class_idx[split_point:])

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def stratified_subset(
    images: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a fixed-seed stratified subset — same call, same subset, every time."""
    rng = np.random.default_rng(seed)
    selected_idx = []
    for label_idx in range(config.NUM_CLASSES):
        class_idx = np.where(labels == label_idx)[0]
        selected_idx.append(
            rng.choice(class_idx, size=samples_per_class, replace=False)
        )

    selected_idx = np.concatenate(selected_idx)
    rng.shuffle(selected_idx)
    return images[selected_idx], labels[selected_idx]


def get_overfit_subset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The fixed small train/val subset used to deliberately overfit.

    Reproducible across every script via config.RANDOM_SEED: 500 train images,
    200 val images, drawn from disjoint pools so there is zero leakage.
    """
    train_images, train_labels = load_split(config.TRAIN_DIR)
    (pool_train_x, pool_train_y), (pool_val_x, pool_val_y) = disjoint_split(
        train_images, train_labels, val_fraction=0.2, seed=config.RANDOM_SEED
    )
    x_train, y_train = stratified_subset(
        pool_train_x,
        pool_train_y,
        config.OVERFIT_SAMPLES_PER_CLASS,
        seed=config.RANDOM_SEED,
    )
    x_val, y_val = stratified_subset(
        pool_val_x,
        pool_val_y,
        config.OVERFIT_VAL_SAMPLES_PER_CLASS,
        seed=config.RANDOM_SEED + 1,
    )
    return x_train, y_train, x_val, y_val


def get_full_train_val(
    val_fraction: float = 0.2, seed: int = config.RANDOM_SEED
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Full (non-subsampled) disjoint train/val split — used by the LR range test,
    which needs many batches in a single epoch, not the tiny overfit subset."""
    train_images, train_labels = load_split(config.TRAIN_DIR)
    return disjoint_split(
        train_images, train_labels, val_fraction=val_fraction, seed=seed
    )


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale uint8 [0, 255] images to float32 [0, 1]."""
    return images.astype("float32") / 255.0
