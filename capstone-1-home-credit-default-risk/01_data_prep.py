"""
01_data_prep.py — Capstone 1: Home Credit Default Risk

Loads application_train.csv, validates it against known properties of the
official dataset, handles the DAYS_EMPLOYED sentinel anomaly, checks for
duplicate IDs, and saves a cleaned version for downstream EDA and feature
engineering scripts.
"""

# %% Imports
from pathlib import Path

import config
from utils.data_utils import (
    check_duplicate_ids,
    flag_and_clean_days_employed_anomaly,
    load_application_data,
    validate_application_data,
)

# %% Load raw data
df = load_application_data(config.APPLICATION_TRAIN_PATH)

# %% Validate against known properties of the official dataset
validate_application_data(df, expected_shape=(307511, 122))

# %% Check for duplicate loan IDs (should be 0 — one row per loan)
check_duplicate_ids(df, config.ID_COL)

# %% Handle the DAYS_EMPLOYED sentinel anomaly (365243 == not applicable)
df = flag_and_clean_days_employed_anomaly(df, config.DAYS_EMPLOYED_ANOMALY_VALUE)

# %% Confirm TARGET distribution (sanity check on class imbalance)
target_counts = df[config.TARGET_COL].value_counts(normalize=True)
print(f"[01_data_prep] TARGET distribution:\n{target_counts}")

# %% Save cleaned data for downstream scripts
output_path: Path = config.OUTPUT_DIR / "application_train_clean.parquet"
df.to_parquet(output_path, index=False)
print(f"[01_data_prep] Saved cleaned data to {output_path}")
