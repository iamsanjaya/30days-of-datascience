# ============================================================
# DAY 06 — Out-of-Box Challenge
# Missingness Matrix, Correlation Analysis & MCAR/MAR/MNAR
# Dataset : tlc_combined_clean_jan2023.parquet (from day06_tlc_merge.py)
# Stack   : Pandas, NumPy, Matplotlib, Seaborn
# Run     : AFTER day06_tlc_merge.py has been run
# Output  : ../outputs/missingness_heatmap.png
# ============================================================
#
# CHALLENGE QUESTION:
#   Missing data is rarely random.
#   Can we prove that — using only the data itself?
#
# APPROACH:
#   Build a binary "missingness matrix":
#     1 = value was missing in that row
#     0 = value was present
#   Correlate those binary flags with other numeric columns.
#   High correlation = the missingness has a pattern = MAR.
#   Near-zero correlation = random = MCAR.
#   MNAR cannot be detected statistically — requires domain reasoning.
# ============================================================

# %% Imports & Paths


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "")

# This file depends on the output of day06_tlc_merge.py
CLEAN_FILE = DATA_PROCESSED + "tlc_combined_clean_jan2023.parquet"

sns.set_theme(style="whitegrid")

print("[OK] Imports and paths ready.")
print(f"[LOG] Expecting clean file at: {os.path.abspath(CLEAN_FILE)}")


# Load raw files — missingness analysis must happen BEFORE imputation

green = pd.read_parquet(DATA_RAW + "green_tripdata_2023-01.parquet")
yellow = pd.read_parquet(DATA_RAW + "yellow_tripdata_2023-01.parquet")

green["taxi_type"] = "green"
yellow["taxi_type"] = "yellow"

green = green.rename(
    columns={
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
    }
)
yellow = yellow.rename(
    columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    }
)

green = green.drop(columns=["ehail_fee", "trip_type"], errors="ignore")
yellow = yellow.drop(columns=["airport_fee"], errors="ignore")

shared_cols = sorted(set(green.columns) & set(yellow.columns))
df = pd.concat([green[shared_cols], yellow[shared_cols]], ignore_index=True)

print(f"[RESULT] Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"[LOG] Columns   : {sorted(df.columns.tolist())}")


# %% Identify Columns With Nulls
# After imputation in day06_tlc_merge.py, some nulls may
# still remain (tip_rate and speed_mph are np.nan by design
# when fare or duration is zero — those are valid NaNs, not
# missing data in the MCAR/MAR sense).
# We include them here because the challenge is about
# studying ALL missingness patterns, including engineered ones.


null_counts = df.isnull().sum()
null_pct = (df.isnull().mean() * 100).round(2)

null_summary = pd.DataFrame(
    {
        "null_count": null_counts,
        "null_%": null_pct,
    }
).sort_values("null_%", ascending=False)

null_cols = null_summary[null_summary["null_count"] > 0].index.tolist()

print("=== COLUMNS WITH NULLS ===")
print(null_summary[null_summary["null_count"] > 0].to_string())
print(f"\n[RESULT] {len(null_cols)} columns have nulls: {null_cols}")


# %% Build the Missingness Matrix

# CONCEPT: Convert each null column into a binary flag.
#   1 = the value was missing in that row
#   0 = the value was present

# This turns an absence into a measurable variable.
# We can now run standard correlation against it.

# Column naming convention: MISS_{original_column_name}


miss_matrix = df[null_cols].isnull().astype(int)
miss_matrix.columns = [f"MISS_{col}" for col in null_cols]

print(f"[RESULT] Missingness matrix shape: {miss_matrix.shape}")
print("\n[LOG] First 5 rows of missingness matrix:")
print(miss_matrix.head().to_string())

print("\n[RESULT] Missingness rate per flag column:")
miss_rates = (miss_matrix.mean() * 100).round(2)
print(miss_rates.to_string())


# %% Select Numeric Predictors

# We correlate missingness flags against numeric columns
# that have no nulls themselves (otherwise correlation is
# computed on unequal sample sizes and becomes unreliable).


all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
predictor_cols = [col for col in all_numeric if col not in null_cols]

print(f"[RESULT] Numeric predictor columns ({len(predictor_cols)}):")
print(predictor_cols)


# %% Compute Missingness Correlation Matrix

# CONCEPT: Pearson correlation between a binary variable (0/1)
# and a continuous variable is a point-biserial correlation.
# Pandas .corr() computes this correctly — no special handling needed.

# Interpretation:
#   r near 0    → missingness is unrelated to this predictor → MCAR signal
#   r != 0      → missingness is related to this predictor   → MAR signal
#   |r| > 0.30  → strong MAR — use predictor in imputation model


miss_flag_cols = miss_matrix.columns.tolist()

analysis_df = pd.concat(
    [df[predictor_cols].reset_index(drop=True), miss_matrix.reset_index(drop=True)],
    axis=1,
)

corr_full = analysis_df.corr(numeric_only=True)
corr_matrix = corr_full[miss_flag_cols].loc[predictor_cols]

print("[RESULT] Missingness correlation matrix (predictors × flags):")
print(corr_matrix.round(3).to_string())


# %% Visualise: Missingness Correlation Heatmap


fig, ax = plt.subplots(figsize=(max(8, len(miss_flag_cols) * 2), 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Pearson Correlation (point-biserial)"},
)

ax.set_title(
    "Missingness Correlation Heatmap\n"
    "Rows = numeric predictors | Columns = binary missing flags\n"
    "Non-zero values = MAR pattern detected",
    fontsize=11,
)
ax.set_ylabel("Numeric Predictor Columns")
ax.set_xlabel("Missingness Flag Columns")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "missingness_heatmap.png", dpi=150)
plt.show()
print("[SAVED] missingness_heatmap.png")


# %% MCAR / MAR / MNAR Classification

# CONCEPT — the three types of missingness:

# MCAR (Missing Completely At Random)
#   The probability of being missing is the same for all rows.
#   No predictor correlates with the missing flag.
#   Detection: max |correlation| across all predictors ≈ 0.
#   Action: safe to drop rows or use simple imputation.

# MAR (Missing At Random)
#   The probability of being missing depends on OTHER observed columns,
#   not on the missing value itself.
#   Detection: at least one predictor has |correlation| > threshold.
#   Action: impute using the correlated predictor(s).

# MNAR (Missing Not At Random)
#   The probability of being missing depends on the MISSING VALUE itself.
#   Example: high earners skip the income question.
#   Detection: cannot be proven from data alone — needs domain reasoning.
#   Action: flag as MNAR, document the assumption, handle with caution.

# Thresholds used here (conservative, common in practice):
#   |r| < 0.05  → MCAR
#   |r| 0.05–0.30 → MAR (weak)
#   |r| >= 0.30 → MAR (strong — use in imputation model)


MCAR_THRESHOLD = 0.05
MAR_STRONG_THRESHOLD = 0.30

print("=" * 60)
print("MISSINGNESS CLASSIFICATION REPORT")
print("=" * 60)

classifications = {}

for flag_col in miss_flag_cols:
    raw_col = flag_col.replace("MISS_", "")
    pct_miss = miss_rates[flag_col]
    col_corr = corr_matrix[flag_col].abs()
    max_corr = col_corr.max()
    top_pred = col_corr.idxmax()

    if max_corr < MCAR_THRESHOLD:
        label = "MCAR"
        verdict = "No correlation with observed columns. Safe to drop or simple-impute."
        action = "fillna(median) or dropna()"
    elif max_corr < MAR_STRONG_THRESHOLD:
        label = "MAR (weak)"
        verdict = f"Weak correlation with '{top_pred}' (r={corr_matrix[flag_col][top_pred]:.3f}). Missingness partly explained."
        action = f"fillna(median grouped by {top_pred})"
    else:
        label = "MAR (strong)"
        verdict = f"Strong correlation with '{top_pred}' (r={corr_matrix[flag_col][top_pred]:.3f}). Missingness well-explained."
        action = f"KNNImputer or model-based imputation using {top_pred}"

    classifications[raw_col] = {
        "label": label,
        "pct_miss": pct_miss,
        "max_corr": max_corr,
        "top_pred": top_pred,
        "action": action,
    }

    print(f"\n[{raw_col}]")
    print(f"Classification: {label}")
    print(f"Missing: {pct_miss:.2f}%")
    print(f"Max |r|: {max_corr:.4f}  (with '{top_pred}')")
    print(f"Verdict: {verdict}")
    print(f"Recommended: {action}")

print("\n" + "=" * 60)


# %% Domain Reasoning: MNAR Check

# Statistical tests cannot detect MNAR. We reason from domain knowledge.

# Finding: tip_rate is NaN when fare_amount <= 0 (by our own formula).
#          This is a structural NaN — we created it intentionally.
#          It is neither MCAR nor MAR; it is a feature design decision.

# Finding: speed_mph is NaN when trip_duration_min <= 0.
#          Same reasoning — structural NaN.

# Finding: In the original raw data (before imputation in Cell 13 of
#          day06_tlc_merge.py), passenger_count, RatecodeID,
#          congestion_surcharge, store_and_fwd_flag, and payment_type
#          are ALL missing for the exact same 6.3% of green rows.
#          This cluster pattern indicates a vendor-level reporting failure —
#          Vendor 1 (VeriFone) does not submit these fields for certain trips.
#          That is MAR on VendorID, not MCAR.

# Potential MNAR case: payment_type.
#          Cash payments are less likely to have digital records.
#          If payment_type is missing precisely because the trip was paid cash,
#          then missingness depends on payment_type itself — MNAR.
#          We cannot confirm this from data; we flag it for domain review.


print("=== DOMAIN REASONING: MNAR CANDIDATES ===\n")
print("[tip_rate]")
print("  Structural NaN — created by formula when fare_amount <= 0.")
print("  Not a data quality issue. Expected and correct.\n")

print("[speed_mph]")
print("  Structural NaN — created by formula when trip_duration_min <= 0.")
print("  Not a data quality issue. Expected and correct.\n")

print("[payment_type — MNAR candidate]")
print("  Cash payments may not generate digital records.")
print("  If missingness depends on whether the trip was cash, this is MNAR.")
print("  Statistical test cannot confirm. Flag for domain expert review.\n")

print("[Cluster missingness in green (vendor pattern)]")
print("  passenger_count, RatecodeID, congestion_surcharge, store_and_fwd_flag")
print("  are missing together for the same 6.3% of green rows.")
print("  Root cause: VendorID 1 does not report these fields for certain trips.")
print("  Classification: MAR on VendorID — not MCAR.")


# %% Summary


print("\n" + "=" * 60)
print("DAY 06 — OUT-OF-BOX CHALLENGE COMPLETE")
print("=" * 60)
print(f"Columns analysed: {len(null_cols)}")
print(f"Missingness flags built: {len(miss_flag_cols)}")
print(f"Predictor columns used: {len(predictor_cols)}")
print(f"Heatmap saved to: {os.path.abspath(OUTPUTS_DIR)}missingness_heatmap.png")
print("=" * 60)
print("\nKey takeaway:")
print("Missing data is almost never random in real-world datasets.")
print("Always ask WHY before filling.")
print("The missing value's pattern is itself a signal.")
