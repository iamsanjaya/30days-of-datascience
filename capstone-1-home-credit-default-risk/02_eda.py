"""
02_eda.py — Capstone 1: Home Credit Default Risk

Targeted EDA — not an exhaustive dump of every column, but 4 specific
questions chosen because they go beyond what every public Home Credit
notebook already covers:

1. Does the DAYS_EMPLOYED anomaly flag (pensioner/unemployed-equivalent)
   correlate with default rate?
2. Does missingness in EXT_SOURCE_1/2/3 itself carry signal about default
   risk, independent of the score values themselves?
3. Does the credit-to-income ratio show a clear relationship with default,
   or is it noisier than commonly assumed?
4. [Your own original finding — see placeholder at the bottom]

Each chart title states a conclusion, not a description (Day 7 convention).
"""

# %% Imports
import matplotlib.pyplot as plt
import pandas as pd

import config
from utils.plot_utils import save_and_close

EDA_OUTPUT_DIR = config.OUTPUT_DIR / "eda"

# %% Load cleaned data from 01_data_prep.py
df = pd.read_parquet(config.OUTPUT_DIR / "application_train_clean.parquet")
print(f"[02_eda] Loaded {df.shape[0]} rows, {df.shape[1]} columns")

baseline_default_rate = df[config.TARGET_COL].mean()
print(f"[02_eda] Overall default rate: {baseline_default_rate:.2%}")

# %% Finding 1 — DAYS_EMPLOYED anomaly vs default rate
anom_default_rates = df.groupby("DAYS_EMPLOYED_ANOM")[config.TARGET_COL].mean()
anom_group_sizes = df["DAYS_EMPLOYED_ANOM"].value_counts()

print("\n[Finding 1] Default rate by DAYS_EMPLOYED_ANOM:")
print(anom_default_rates)
print(f"Group sizes:\n{anom_group_sizes}")

fig, ax = plt.subplots(figsize=(6, 5))
anom_default_rates.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.axhline(baseline_default_rate, color="gray", linestyle="--", label="Overall avg")
ax.set_xticklabels(
    ["Normal DAYS_EMPLOYED", "Anomalous (pensioner/unemployed)"], rotation=0
)
ax.set_ylabel("Default rate")
direction = (
    "higher" if anom_default_rates[True] > anom_default_rates[False] else "lower"
)
ax.set_title(f"Pensioner/Unemployed Applicants Default at a {direction.title()} Rate")
ax.legend()
save_and_close(fig, EDA_OUTPUT_DIR, "01_days_employed_anomaly_vs_default.png")

# %% Finding 2 — EXT_SOURCE missingness vs default rate
ext_source_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
missingness_results = {}

for col in ext_source_cols:
    is_missing = df[col].isnull()
    pct_missing = is_missing.mean()
    default_rate_missing = df.loc[is_missing, config.TARGET_COL].mean()
    default_rate_present = df.loc[~is_missing, config.TARGET_COL].mean()
    missingness_results[col] = {
        "pct_missing": pct_missing,
        "default_rate_if_missing": default_rate_missing,
        "default_rate_if_present": default_rate_present,
    }
    print(
        f"\n[Finding 2] {col}: {pct_missing:.1%} missing | "
        f"default rate if missing: {default_rate_missing:.2%} | "
        f"default rate if present: {default_rate_present:.2%}"
    )

missingness_df = pd.DataFrame(missingness_results).T
fig, ax = plt.subplots(figsize=(8, 5))
missingness_df[["default_rate_if_missing", "default_rate_if_present"]].plot(
    kind="bar", ax=ax
)
ax.axhline(baseline_default_rate, color="gray", linestyle="--", label="Overall avg")
ax.set_ylabel("Default rate")
ax.set_title("EXT_SOURCE Missingness Itself Predicts Default Risk")
ax.legend(["Missing", "Present", "Overall avg"])
save_and_close(fig, EDA_OUTPUT_DIR, "02_ext_source_missingness_vs_default.png")

# %% Finding 3 — Credit-to-income ratio vs default rate
df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

# Quantile bins rather than fixed cutoffs — lets the data's own distribution
# define "low/medium/high" rather than an arbitrary threshold
df["CREDIT_INCOME_RATIO_BIN"] = pd.qcut(
    df["CREDIT_INCOME_RATIO"], q=10, duplicates="drop"
)
ratio_default_rates = df.groupby("CREDIT_INCOME_RATIO_BIN", observed=True)[
    config.TARGET_COL
].mean()

print(
    f"\n[Finding 3] Default rate by credit/income ratio decile:\n{ratio_default_rates}"
)

fig, ax = plt.subplots(figsize=(9, 5))
ratio_default_rates.plot(kind="bar", ax=ax, color="#55A868")
ax.axhline(baseline_default_rate, color="gray", linestyle="--", label="Overall avg")
ax.set_ylabel("Default rate")
ax.set_xlabel("Credit-to-income ratio decile (low → high)")
peak_decile_idx = ratio_default_rates.argmax()
is_peak_in_middle = 0 < peak_decile_idx < len(ratio_default_rates) - 1
if is_peak_in_middle:
    title = "Default Risk Peaks at Moderate Leverage, Not Extreme Leverage"
elif peak_decile_idx == len(ratio_default_rates) - 1:
    title = "Default Risk Rises Steadily with Credit-to-Income Ratio"
else:
    title = "Default Risk Falls with Credit-to-Income Ratio"
ax.set_title(title)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend()
save_and_close(fig, EDA_OUTPUT_DIR, "03_credit_income_ratio_vs_default.png")

# %% Finding 4 — Is the DAYS_EMPLOYED anomaly just a pensioner proxy?
#
# Finding 1 showed anomalous applicants default LESS, not more (5.4% vs
# 8.66%) — counter to the naive "unemployed = riskier" assumption. This
# checks whether that's really about pension income stability, by seeing
# how much the anomaly flag overlaps with NAME_INCOME_TYPE == "Pensioner",
# and whether default rate varies meaningfully across income types more
# broadly (not just the binary anomaly flag).

anomaly_income_type_crosstab = pd.crosstab(
    df["DAYS_EMPLOYED_ANOM"], df["NAME_INCOME_TYPE"], normalize="index"
)
print(
    f"\n[Finding 4] Income type composition by anomaly flag:\n{anomaly_income_type_crosstab}"
)

pct_anomaly_that_is_pensioner = anomaly_income_type_crosstab.loc[True, "Pensioner"]
print(
    f"[Finding 4] {pct_anomaly_that_is_pensioner:.1%} of anomalous-DAYS_EMPLOYED "
    "applicants are Pensioners."
)

income_type_default_rates = (
    df.groupby("NAME_INCOME_TYPE")[config.TARGET_COL].mean().sort_values()
)
print(f"\n[Finding 4] Default rate by NAME_INCOME_TYPE:\n{income_type_default_rates}")

income_type_counts = df["NAME_INCOME_TYPE"].value_counts()
print(f"\n[Finding 4] NAME_INCOME_TYPE group sizes:\n{income_type_counts}")

# Some categories (Businessman, Student) have vanishingly few rows in this
# dataset — a 0% default rate there is a sample-size artifact, not a real
# finding. Exclude unreliable tiny groups from the headline comparison.
MIN_RELIABLE_GROUP_SIZE = 50
reliable_types = income_type_counts[income_type_counts >= MIN_RELIABLE_GROUP_SIZE].index
excluded_types = income_type_counts[income_type_counts < MIN_RELIABLE_GROUP_SIZE].index
if len(excluded_types) > 0:
    print(
        f"[Finding 4] Excluding from headline comparison (n < {MIN_RELIABLE_GROUP_SIZE}): "
        f"{list(excluded_types)}"
    )

reliable_rates = income_type_default_rates.loc[
    income_type_default_rates.index.isin(reliable_types)
]

fig, ax = plt.subplots(figsize=(9, 5))
income_type_default_rates.plot(kind="barh", ax=ax, color="#8172B2")
ax.axvline(baseline_default_rate, color="gray", linestyle="--", label="Overall avg")
ax.set_xlabel("Default rate")
lowest_risk_type = reliable_rates.index[0]
highest_risk_type = reliable_rates.index[-1]
risk_multiple = reliable_rates.iloc[-1] / reliable_rates.iloc[0]
ax.set_title(
    f"{highest_risk_type} Applicants Default {risk_multiple:.1f}x "
    f"More Often Than {lowest_risk_type} Applicants (n≥{MIN_RELIABLE_GROUP_SIZE} groups only)"
)
ax.legend()
save_and_close(fig, EDA_OUTPUT_DIR, "04_income_type_vs_default.png")

print("\n[02_eda] All 4 findings complete. Saved 4 PNGs to outputs/eda/.")
