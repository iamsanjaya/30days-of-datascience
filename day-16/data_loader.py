# data_loader.py — Day 16
# Responsible for loading the Ames Housing CSV and returning a clean split.
# Feature engineering is intentionally kept lightweight here — Day 14 did the
# deep work. Day 16's focus is tuning, not re-engineering.

import pandas as pd
from sklearn.model_selection import train_test_split

from config import TRAIN_CSV, TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_ames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load Ames Housing, drop ID + low-signal columns, return train/val split.

    Returns
    -------
    X_train, X_val, y_train, y_val : DataFrames / Series
    """
    df = pd.read_csv(TRAIN_CSV)

    # Drop the row identifier — carries no predictive signal.
    df.drop(columns=["Id"], inplace=True, errors="ignore")

    # Separate target before any transformation.
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    # ── Minimal cleaning ──────────────────────────────────────────────────────
    # Drop columns with more than 40 % missing values.
    high_missing = X.columns[X.isnull().mean() > 0.40].tolist()
    X.drop(columns=high_missing, inplace=True)

    # Numeric columns: fill remaining NaNs with median.
    num_cols = X.select_dtypes(include="number").columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Categorical columns: fill NaNs with the literal string "Missing".
    cat_cols = X.select_dtypes(include="object").columns
    X[cat_cols] = X[cat_cols].fillna("Missing")

    # One-hot encode all categoricals so XGBoost receives a numeric matrix.
    X = pd.get_dummies(X, columns=cat_cols.tolist(), drop_first=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"[data_loader] Train: {X_train.shape} | Val: {X_val.shape}")
    print(f"[data_loader] Dropped high-missing cols: {high_missing}")
    return X_train, X_val, y_train, y_val
