"""
utils/data.py
Loading the raw Kaggle CSV, mapping labels to integer ids, and producing a
stratified train/val/test split. No synthetic rows are ever generated here —
if the raw file is missing, we fail loudly instead of fabricating data.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def load_raw_tweets(path: Path = config.RAW_CSV_PATH) -> pd.DataFrame:
    """Load the raw Kaggle Twitter US Airline Sentiment CSV.

    Raises FileNotFoundError with a clear message rather than silently
    proceeding — there is no fallback synthetic dataset in this project.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'Twitter US Airline Sentiment' from Kaggle "
            "(crowdflower/twitter-airline-sentiment), then place the CSV "
            f"(rename to '{config.RAW_CSV_NAME}') at: {path}"
        )

    df = pd.read_csv(path)

    missing = {config.TEXT_COLUMN, config.LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found in {path}. "
            f"Columns present: {list(df.columns)}"
        )

    return df


def clean_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing text/label, map label strings to ids, dedupe."""
    df = df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN]).copy()
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN]).copy()

    unmapped = set(df[config.LABEL_COLUMN].unique()) - set(config.LABEL_MAPPING)
    if unmapped:
        raise ValueError(
            f"Found label values not in config.LABEL_MAPPING: {unmapped}. "
            "Update config.LABEL_MAPPING / config.CLASS_NAMES to match the raw data."
        )

    df["label_id"] = df[config.LABEL_COLUMN].map(config.LABEL_MAPPING)
    df = df.reset_index(drop=True)
    return df[[config.TEXT_COLUMN, config.LABEL_COLUMN, "label_id"]]


def stratified_split(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, stratified on label_id at both splits."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["label_id"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train.to_csv(config.TRAIN_PATH, index=False)
    val.to_csv(config.VAL_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed splits written by 01_data_preparation.py."""
    for p in (config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run 01_data_preparation.py first.")
    return (
        pd.read_csv(config.TRAIN_PATH),
        pd.read_csv(config.VAL_PATH),
        pd.read_csv(config.TEST_PATH),
    )


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Class counts, indexed by CLASS_NAMES order — useful for the README + journal."""
    counts = df["label_id"].value_counts().sort_index()
    counts.index = [config.CLASS_NAMES[i] for i in counts.index]
    return counts
