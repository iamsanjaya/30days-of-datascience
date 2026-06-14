# %%
"""
03_random_forest_comparison.py

This file is reserved for extended Random Forest analysis as the day-10
project grows. Currently the DT vs RF comparison lives in 01_ (CV scores,
feature importances, probability calibration). Anything that warrants its
own deep-dive — e.g. OOB error analysis, feature importance stability
across random seeds, or a proper train/val/test holdout comparison —
belongs here.

Run order: 01_ → 02_ → 03_
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_RAW,
    OUTPUTS_DIR,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    BINARY_COLS,
    YES_NO_COLS,
    MULTI_COLS,
    TARGET_COL,
)

import os
import warnings

warnings.filterwarnings("ignore")
# %%
# LOAD & PREPROCESS (same pipeline as 01_ and 02_)

df_raw = pd.read_csv(DATA_RAW)
df = df_raw.copy()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop(columns=["customerID"])

for col in BINARY_COLS:
    df[col] = (df[col] == "Yes").astype(int)
for col in YES_NO_COLS:
    df[col] = (df[col] == "Yes").astype(int)

le = LabelEncoder()
for col in MULTI_COLS:
    df[col] = np.array(le.fit_transform(df[col]), dtype=int)

df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]
X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"[RESULT] Data ready: {X.shape[0]:,} rows × {X.shape[1]} features")

# %%
# OOB ERROR vs N_ESTIMATORS
# Out-of-bag error is a free internal CV estimate available in Random Forests.
# Each tree is trained on a bootstrap sample; the ~37% of rows not in that
# sample (the "out-of-bag" rows) are used to estimate generalisation error
# without needing a separate validation set.
# Watching OOB error stabilise tells you the minimum n_estimators you need.

print("\n[STEP] OOB error vs number of trees...")

estimator_range = [10, 25, 50, 75, 100, 150, 200, 300]
oob_errors = []

for n in estimator_range:
    rf_oob = RandomForestClassifier(
        n_estimators=n,
        max_depth=RF_MAX_DEPTH,
        oob_score=True,  # enable OOB estimation
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_oob.fit(X, y)
    oob_errors.append(1 - rf_oob.oob_score_)
    print(f"  n_estimators={n:>4}  OOB error={oob_errors[-1]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(estimator_range, oob_errors, marker="o", color="seagreen", linewidth=2)
ax.set_xlabel("Number of Trees (n_estimators)")
ax.set_ylabel("OOB Error Rate  (1 − OOB accuracy)")
ax.set_title(
    "OOB Error vs n_estimators — Telco Churn\n"
    "Error stabilises: adding more trees beyond this point gives diminishing returns"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
oob_path = os.path.join(OUTPUTS_DIR, "rf_oob_error_vs_n_estimators.png")
plt.savefig(oob_path, dpi=150)
plt.close()
print(f"\n[RESULT] OOB error plot saved → {oob_path}")

# %%
# FEATURE IMPORTANCE STABILITY ACROSS RANDOM SEEDS
# A single RF's feature importances can shift with different random seeds.
# Running multiple seeds and measuring the standard deviation of each
# feature's importance tells you which signals are robust vs noisy.

print("\n[STEP] Feature importance stability across 10 random seeds...")

seeds = list(range(10))
importance_matrix = []

for seed in seeds:
    rf_seed = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    rf_seed.fit(X, y)
    importance_matrix.append(rf_seed.feature_importances_)

imp_df = pd.DataFrame(importance_matrix, columns=FEATURE_COLS)
imp_mean = imp_df.mean().sort_values(ascending=False)
imp_std = imp_df.std()

# Keep top 10 features for readability
top10 = imp_mean.head(10).index.tolist()

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    top10,
    np.array(imp_mean[top10].values, dtype=float),
    xerr=np.array(imp_std[top10].values, dtype=float),
    color="seagreen",
    edgecolor="white",
    alpha=0.85,
    capsize=4,
)
ax.set_title(
    "RF Feature Importance Stability — Top 10 Features\n"
    f"Mean ± Std across 10 random seeds  ({RF_N_ESTIMATORS} trees each)",
    fontsize=11,
)
ax.set_xlabel("Mean Decrease in Impurity")
ax.invert_yaxis()
plt.tight_layout()
stability_path = os.path.join(OUTPUTS_DIR, "rf_importance_stability.png")
plt.savefig(stability_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[RESULT] Importance stability plot saved → {stability_path}")

print("\nTop 10 features — mean importance ± std across 10 seeds:")
for feat in top10:
    print(f"  {feat:<25}  {imp_mean[feat]:.4f}  ±  {imp_std[feat]:.4f}")

# %%
# SUMMARY

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DAY 10 — FILE 03 COMPLETE                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║  OOB Error Analysis                                                      ║
║    Minimum OOB error : {min(oob_errors):.4f}  (at n={estimator_range[oob_errors.index(min(oob_errors))]} trees)                          ║
║    OOB stabilises well before {RF_N_ESTIMATORS} trees on this dataset                  ║
║                                                                          ║
║  Feature Importance Stability                                            ║
║    Most stable feature : {imp_std[top10].idxmin():<25}(std={imp_std[imp_std.index.isin(top10)].min():.4f})           ║
║    Least stable feature: {imp_std[top10].idxmax():<25}(std={imp_std[imp_std.index.isin(top10)].max():.4f})           ║
║                                                                          ║
║  Outputs                                                                 ║
║    rf_oob_error_vs_n_estimators.png                                      ║
║    rf_importance_stability.png                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
