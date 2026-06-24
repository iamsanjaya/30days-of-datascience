"""
Data utilities: scan the Microsoft Cats vs Dogs PetImages/Cat,
PetImages/Dog folders, validate every file, build a stratified
train/val/test split, and construct tf.data.Dataset pipelines with
backbone-appropriate preprocessing and (optional) augmentation.

On corrupt/incompatible files: this dataset ships with a number of
files that look fine to PIL but that tf.io.decode_image (the op
actually used at training time) rejects — zero-byte files, files with
a .jpg extension that are really BMP/TIFF/PSD internally, and PNGs
with an unsupported channel count (e.g. 2-channel grayscale+alpha).
Trying to infer each of these categories individually via PIL
heuristics didn't converge — there were more edge cases than
categories checked. validate_images() below instead calls
tf.io.decode_image directly, eagerly, on every file: the same op
build_dataset() uses inside the tf.data graph. If it raises here, it
would have raised during training; excluding it now is authoritative,
not a guess.

Separately: neither tf.io.decode_image nor Keras's ImageDataGenerator
skips bad files gracefully at training time — both raise mid-read, the
only difference being *when* you find out. ImageDataGenerator finds
out mid-epoch, partway through training, because it doesn't validate
ahead of time; you lose whatever compute ran before the crash. The fix
here is to validate every file once, up front, in 01_prepare_data.py,
and exclude bad files from the manifest entirely — so training never
touches them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


def scan_raw_directory(raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """
    Scan the PetImages/Cat, PetImages/Dog folder structure and return a
    DataFrame with columns: filepath, label (0=cat, 1=dog).
    """
    cat_dir = raw_dir / config.CAT_DIR_NAME
    dog_dir = raw_dir / config.DOG_DIR_NAME

    if not cat_dir.exists() or not dog_dir.exists():
        raise FileNotFoundError(
            f"Expected folders not found: {cat_dir} and/or {dog_dir}\n"
            "Download the Microsoft Cats vs Dogs dataset "
            "(kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset), "
            f"extract it, and place PetImages/ at: {raw_dir}"
        )

    cat_files = sorted(cat_dir.glob("*.jpg"))
    dog_files = sorted(dog_dir.glob("*.jpg"))

    if not cat_files or not dog_files:
        raise FileNotFoundError(
            f"No .jpg files found under {cat_dir} or {dog_dir}. "
            "Check that the dataset was extracted correctly."
        )

    records = [(str(p), 0) for p in cat_files] + [(str(p), 1) for p in dog_files]
    df = pd.DataFrame(records, columns=["filepath", "label"])
    return df


def validate_images(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate every file by actually running it through tf.io.decode_image
    (eager mode, same op build_dataset() uses inside tf.data), and drop
    anything that raises. Returns (clean_df, removed_df) — removed_df is
    saved as a log so the exclusion is documented, not silent.

    This replaced an earlier PIL-heuristic version (checking format
    strings and band counts) that kept missing edge cases one at a
    time — PIL can successfully open files that TensorFlow's decoder
    still rejects, for reasons that don't reduce to a short checklist.
    Calling the real decoder directly is authoritative instead of a
    guess.
    """
    good_rows = []
    bad_rows = []

    for _, row in df.iterrows():
        path = Path(row["filepath"])
        reason = None

        if path.stat().st_size == 0:
            reason = "zero_byte_file"
        else:
            try:
                raw = tf.io.read_file(str(path))
                image = tf.io.decode_image(
                    raw, channels=config.CHANNELS, expand_animations=False
                )
                # decode_image can defer some of the actual decode work;
                # touching .shape and converting to numpy forces it to
                # fully materialize, same as it will inside the training
                # graph.
                _ = image.shape
                _ = image.numpy()
            except (
                Exception
            ) as e:  # noqa: BLE001 — any decode failure is disqualifying here
                reason = f"undecodable_by_tf: {type(e).__name__}"

        if reason is None:
            good_rows.append(row)
        else:
            bad_rows.append({**row, "reason": reason})

    clean_df = pd.DataFrame(good_rows).reset_index(drop=True)
    removed_df = pd.DataFrame(bad_rows).reset_index(drop=True)
    return clean_df, removed_df


def save_corrupt_files_log(removed_df: pd.DataFrame) -> None:
    config.DATA_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    removed_df.to_csv(config.CORRUPT_FILES_LOG, index=False)


def make_splits(
    df: pd.DataFrame,
    sample_size: int | None = config.SAMPLE_SIZE,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Optionally subsample, then produce stratified train/val/test splits.
    Fractions must sum to 1.0 (within floating point tolerance).
    """
    assert (
        abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6
    ), "train/val/test fractions must sum to 1.0"

    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["label"],
            random_state=seed,
        )

    train_df, holdout_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["label"],
        random_state=seed,
    )
    val_relative_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=val_relative_frac,
        stratify=holdout_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(config.VAL_SPLIT_CSV, index=False)
    test_df.to_csv(config.TEST_SPLIT_CSV, index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (config.TRAIN_SPLIT_CSV, config.VAL_SPLIT_CSV, config.TEST_SPLIT_CSV):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 01_prepare_data.py first.")
    train_df = pd.read_csv(config.TRAIN_SPLIT_CSV)
    val_df = pd.read_csv(config.VAL_SPLIT_CSV)
    test_df = pd.read_csv(config.TEST_SPLIT_CSV)
    return train_df, val_df, test_df


def _preprocess_input(image: tf.Tensor) -> tf.Tensor:
    """Dispatch to the correct backbone preprocessing. ResNet50 expects
    caffe-style (BGR, mean-subtracted) input; Xception expects images
    scaled to [-1, 1]. Mixing these up silently tanks accuracy without
    erroring, so this is centralized in one place rather than duplicated."""
    if config.BACKBONE == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input(image)
    elif config.BACKBONE == "xception":
        return tf.keras.applications.xception.preprocess_input(image)
    else:
        raise ValueError(f"Unknown backbone in config: {config.BACKBONE}")


def load_and_preprocess_single(
    filepath: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Public single-example loader, reused by gradcam analysis to build
    the exact same preprocessed tensor the model was trained on."""
    return _load_and_preprocess(filepath, label)


def _load_and_preprocess(
    filepath: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(filepath)
    # decode_image (not decode_jpeg) auto-detects the real format —
    # this dataset has a number of files with a .jpg extension that
    # are actually BMP/PNG/GIF internally, which decode_jpeg can't
    # handle and decode_image can. expand_animations=False forces a
    # static (no-frame-dimension) output even for the rare GIF.
    image = tf.io.decode_image(image, channels=config.CHANNELS, expand_animations=False)
    image.set_shape([None, None, config.CHANNELS])
    image = tf.image.resize(image, config.IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = _preprocess_input(image)
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def build_dataset(
    df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """
    Build a batched, prefetched tf.data.Dataset from a (filepath, label)
    DataFrame. Augmentation is applied only when explicitly requested
    (i.e. only for the training set).
    """
    filepaths = df["filepath"].to_numpy(dtype=str)
    labels = df["label"].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
