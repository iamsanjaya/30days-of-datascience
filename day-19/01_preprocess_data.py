"""
Day 19 — Step 1: Load Adult Census Income data, clean it, encode it,
and produce a reproducible train/val/test split.

Run: python 01_preprocess_data.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from numpy.typing import NDArray
from typing import cast

import config


def load_raw_data() -> pd.DataFrame:
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected raw data at {config.RAW_DATA_PATH}. "
            "Download 'adult.csv' from Kaggle (uciml/adult-census-income) and place it there."
        )
    df = pd.read_csv(config.RAW_DATA_PATH)
    df.replace(config.MISSING_TOKEN, np.nan, inplace=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, config.NUMERIC_COLS),
            ("categorical", categorical_pipeline, config.CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def main():
    df = load_raw_data()
    print(f"Loaded raw data: {df.shape}")

    y = (df[config.TARGET_COL].str.strip() == config.POSITIVE_LABEL).astype(int)
    X = df[config.NUMERIC_COLS + config.CATEGORICAL_COLS]

    # First split off test set, then split remainder into train/val
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    X_train_proc = cast(
        NDArray[np.float64],
        preprocessor.fit_transform(X_train),
    )

    X_val_proc = cast(
        NDArray[np.float64],
        preprocessor.transform(X_val),
    )

    X_test_proc = cast(
        NDArray[np.float64],
        preprocessor.transform(X_test),
    )
    np.savez(
        config.PROCESSED_DIR / "splits.npz",
        X_train=X_train_proc,
        y_train=y_train.to_numpy(),
        X_val=X_val_proc,
        y_val=y_val.to_numpy(),
        X_test=X_test_proc,
        y_test=y_test.to_numpy(),
    )

    print(
        f"Train: {X_train_proc.shape}, Val: {X_val_proc.shape}, Test: {X_test_proc.shape}"
    )
    print(
        f"Positive class rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}"
    )
    print(f"Saved processed splits to {config.PROCESSED_DIR / 'splits.npz'}")
    print(f"Saved preprocessor to {config.PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
