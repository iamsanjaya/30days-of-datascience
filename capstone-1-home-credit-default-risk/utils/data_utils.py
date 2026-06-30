"""
Shared utilities for loading, validating, and cleaning the Home Credit
application data. Used by 01_data_prep.py and downstream scripts.
"""

from pathlib import Path
from typing import Dict, cast

import pandas as pd


def load_application_data(path: Path) -> pd.DataFrame:
    """Load application_train.csv (or application_test.csv) from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file at {path}, but it doesn't exist. "
            "Check config.RAW_DATA_DIR and confirm the file was downloaded."
        )
    return pd.read_csv(path)


def validate_application_data(
    df: pd.DataFrame,
    expected_shape: tuple[int, int] | None = None,
) -> None:
    """
    Sanity-check the loaded dataframe against known properties of the
    official dataset. Raises AssertionError on mismatch rather than
    silently continuing with unexpected data.
    """
    if expected_shape is not None:
        assert df.shape == expected_shape, (
            f"Shape mismatch: got {df.shape}, expected {expected_shape}. "
            "Confirm you downloaded the correct file from the official "
            "competition page, not a re-upload with different columns."
        )

    dtype_counts: Dict[str, int] = cast(
        Dict[str, int], df.dtypes.value_counts().to_dict()
    )
    print(f"[validate_application_data] Shape: {df.shape}")
    print(f"[validate_application_data] Dtype counts: {dtype_counts}")


def check_duplicate_ids(df: pd.DataFrame, id_col: str) -> int:
    """Return count of duplicate IDs. Should be 0 — one row per loan."""
    n_duplicates = int(df[id_col].duplicated().sum())
    if n_duplicates > 0:
        print(
            f"[check_duplicate_ids] WARNING: {n_duplicates} duplicate "
            f"{id_col} values found."
        )
    else:
        print(f"[check_duplicate_ids] OK: no duplicate {id_col} values.")
    return n_duplicates


def flag_and_clean_days_employed_anomaly(
    df: pd.DataFrame, anomaly_value: int
) -> pd.DataFrame:
    """
    DAYS_EMPLOYED contains a sentinel value (365243) for applicants who are
    pensioners/unemployed — not a real value of ~1000 years employed.

    Adds a boolean flag column preserving the fact that this was anomalous
    (since that itself may carry signal — e.g. pensioner status), then
    replaces the sentinel with NaN so it doesn't distort later statistics
    or models.
    """
    df = df.copy()
    is_anomaly = df["DAYS_EMPLOYED"] == anomaly_value
    n_anomalies = int(is_anomaly.sum())
    print(
        f"[flag_and_clean_days_employed_anomaly] Found {n_anomalies} "
        f"anomalous DAYS_EMPLOYED values ({n_anomalies / len(df):.1%} of rows)."
    )
    df["DAYS_EMPLOYED_ANOM"] = is_anomaly
    df.loc[is_anomaly, "DAYS_EMPLOYED"] = pd.NA
    return df
