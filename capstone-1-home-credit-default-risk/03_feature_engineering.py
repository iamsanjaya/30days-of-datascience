"""
03_feature_engineering.py — Capstone 1: Home Credit Default Risk

Creates deterministic, row-wise engineered features informed directly by
the EDA findings in 02_eda.py:

  - Finding 1/4: DAYS_EMPLOYED_ANOM is predictive (already in the cleaned
    data from 01_data_prep.py) — kept as a feature here, not just a
    cleaning artifact.
  - Finding 2: EXT_SOURCE_1/2/3 missingness itself carries signal — flags
    created here. The actual NaN values are left in place; *imputing*
    them requires a fitted statistic (e.g. median), which belongs in the
    Pipeline in 04_train_baseline_models.py to avoid leakage.
  - Finding 3: CREDIT_INCOME_RATIO promoted from EDA scratch work into a
    real feature.

Deliberately NOT done here (deferred to the Pipeline in 04, post-split):
  - Imputation of any missing values
  - One-hot/categorical encoding
  - Scaling

All of those require fitting a statistic on data, and fitting before the
train/test split risks leaking test-set information into preprocessing —
even though no TARGET value is touched directly.
"""

# %% Imports
import numpy as np
import pandas as pd

import config

# %% Load cleaned data from 01_data_prep.py
df = pd.read_parquet(config.OUTPUT_DIR / "application_train_clean.parquet")
print(f"[03_feature_engineering] Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# %% Age and tenure conversions
# DAYS_BIRTH, DAYS_EMPLOYED, etc. are negative day-counts relative to the
# application date. Flipping sign and converting to years makes these
# interpretable (and DAYS_EMPLOYED already has its anomaly nulled out by
# 01_data_prep.py, so YEARS_EMPLOYED will be NaN for that ~18% of rows —
# correct behavior, not a bug, since the Pipeline will impute it properly).
df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365
df["YEARS_REGISTRATION"] = -df["DAYS_REGISTRATION"] / 365
df["YEARS_ID_PUBLISH"] = -df["DAYS_ID_PUBLISH"] / 365

# %% Financial ratios
# CREDIT_INCOME_RATIO: promoted directly from Finding 3's EDA scratch work.
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

# ANNUITY_INCOME_RATIO: what fraction of income the periodic loan payment
# consumes — a more direct affordability signal than credit/income alone.
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

# CREDIT_TERM_RATIO: AMT_ANNUITY / AMT_CREDIT. Higher ratio implies a
# shorter effective loan term (paying off a larger fraction of the credit
# per period). Named "_RATIO" rather than "_YEARS" since the exact payment
# period isn't confirmed from the column documentation alone — avoid
# asserting a literal time unit we haven't verified.
df["CREDIT_TERM_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

# GOODS_PRICE_CREDIT_RATIO: how much of the credit amount actually covers
# the goods being purchased — a proxy for "extra" credit beyond the
# purchase itself (e.g. cash-out, fees).
df["GOODS_PRICE_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]

# %% Defensive check — division can produce inf where a denominator is 0
ratio_cols = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM_RATIO",
    "GOODS_PRICE_CREDIT_RATIO",
]
for col in ratio_cols:
    n_infinite = int(np.isinf(df[col]).sum())
    if n_infinite > 0:
        print(
            f"[03_feature_engineering] WARNING: {col} has {n_infinite} infinite values — replacing with NaN"
        )
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

# %% EXT_SOURCE missingness flags (Finding 2 — missingness itself is signal)
# The actual EXT_SOURCE_* values are left untouched (still containing NaN).
# Imputing them happens in the Pipeline in 04, fit only on training folds.
for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    df[f"{col}_MISSING"] = df[col].isnull()

# %% Sanity check — confirm new columns exist and print a quick summary
new_cols = [
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "YEARS_REGISTRATION",
    "YEARS_ID_PUBLISH",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM_RATIO",
    "GOODS_PRICE_CREDIT_RATIO",
    "EXT_SOURCE_1_MISSING",
    "EXT_SOURCE_2_MISSING",
    "EXT_SOURCE_3_MISSING",
]
print(f"\n[03_feature_engineering] New feature summary:\n{df[new_cols].describe()}")
print(f"\n[03_feature_engineering] Shape after feature engineering: {df.shape}")

# %% Save — categorical columns and remaining NaNs are still raw/untouched
output_path = config.OUTPUT_DIR / "application_train_features.parquet"
df.to_parquet(output_path, index=False)
print(f"[03_feature_engineering] Saved to {output_path}")
