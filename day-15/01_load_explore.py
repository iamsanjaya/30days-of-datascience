# 01_load_explore.py — Day 15
# Load Ames Housing, verify shape, print missing-value audit,
# then do a train/test split. No modelling yet — just confirming
# the data is understood before it enters a pipeline.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_PATH,
    TARGET,
    DROP_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    OUTPUT_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ── 1. Load ───────────────────────────────────────────────────────────────────
# %%
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Drop identifier columns ────────────────────────────────────────────────
# %%
df = df.drop(columns=DROP_COLS)
print(f"After dropping identifiers: {df.shape}")

# ── 3. Quick dtype audit ──────────────────────────────────────────────────────
# %%
print("\n── Dtype counts ──")
print(df.dtypes.value_counts())

# ── 4. Missing value audit ────────────────────────────────────────────────────
# %%
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
audit = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
audit = audit[audit["missing_count"] > 0].sort_values("missing_pct", ascending=False)

print(f"\n── {len(audit)} columns with missing values ──")
print(audit.to_string())

# ── 5. Note on high-missing categoricals ─────────────────────────────────────
# %%
# Pool QC (99.6%), Misc Feature (96.4%), Alley (93.2%), Fence (80.5%):
# NaN here means "no pool / no alley / no fence" — absence, not a data error.
# most_frequent imputation will fill these with the mode (typically the
# dominant non-null category). For a proper domain treatment you would fill
# with a literal "None" string; most_frequent is a conservative safe choice
# inside a pipeline without domain-level custom transformers.
print("\nNote: high-missing categoricals represent absent features, not data errors.")
print("most_frequent imputation used inside pipeline (conservative default).")

# ── 6. Target distribution ────────────────────────────────────────────────────
# %%
print(f"\n── {TARGET} distribution ──")
print(df[TARGET].describe().round(0))
print(f"Skew: {df[TARGET].skew():.3f}  (> 1 → log-transform worth exploring)")

# ── 7. Feature set confirmation ───────────────────────────────────────────────
# %%
all_model_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
missing_from_df = [c for c in all_model_features if c not in df.columns]
print(f"\nFeatures declared in config : {len(all_model_features)}")
print(f"Features missing from CSV   : {missing_from_df}")

# ── 8. Train / test split ─────────────────────────────────────────────────────
# %%
X = df[all_model_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\nTrain : {X_train.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
print(f"Train target mean : ${y_train.mean():,.0f}")
print(f"Test  target mean : ${y_test.mean():,.0f}")

# ── 9. Plot: Missing value audit ──────────────────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(10, 7))
audit_plot = audit[audit["missing_pct"] > 0].sort_values("missing_pct")
colors = [
    "#d9534f" if p > 50 else "#f0ad4e" if p > 10 else "#5bc0de"
    for p in audit_plot["missing_pct"]
]
ax.barh(audit_plot.index, audit_plot["missing_pct"], color=colors)
ax.set_xlabel("Missing (%)")
ax.set_title(
    "Missing Value Audit — Ames Housing\n" "(red >50% · orange >10% · blue <10%)"
)
ax.axvline(50, color="#d9534f", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(10, color="#f0ad4e", linestyle="--", linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_missing_value_audit.png"), dpi=150)
plt.close()
print("Saved: outputs/01_missing_value_audit.png")

# ── 10. Plot: SalePrice distribution ─────────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y, bins=50, color="#5b8db8", edgecolor="white", linewidth=0.4)
axes[0].set_title("SalePrice — Raw (skew = 1.74)")
axes[0].set_xlabel("SalePrice ($)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(y), bins=50, color="#6aaa64", edgecolor="white", linewidth=0.4)
axes[1].set_title("SalePrice — log1p transformed (near-normal)")
axes[1].set_xlabel("log1p(SalePrice)")
axes[1].set_ylabel("Count")

plt.suptitle("Target Distribution: Raw vs Log-Transformed", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_target_distribution.png"), dpi=150)
plt.close()
print("Saved: outputs/01_target_distribution.png")

print("\n01_load_explore.py complete — data verified, split ready.")
