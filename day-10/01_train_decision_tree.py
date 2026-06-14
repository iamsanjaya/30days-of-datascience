# %%
"""
Standard Task: Decision Trees & Random Forest
Dataset : IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn)
Goal    : Train a Decision Tree (depth 4), visualize it, explain why
          top-of-tree features matter, then train a Random Forest and
          compare both with 5-fold Stratified CV.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_RAW,
    OUTPUTS_DIR,
    RANDOM_STATE,
    DT_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    CV_N_SPLITS,
    BINARY_COLS,
    YES_NO_COLS,
    MULTI_COLS,
    TARGET_COL,
)
import os
import warnings

warnings.filterwarnings("ignore")

# %%
# 1. LOAD
# The file has a .xls extension but is actually CSV.

print("[STEP 1] Loading dataset...")

df_raw = pd.read_csv(DATA_RAW)
print(f"[RESULT] Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"Columns : {df_raw.columns.tolist()}")
print(f"Dtypes  :\n{df_raw.dtypes.to_string()}")

# %%
# 2. CLEAN

print("\n[STEP 2] Cleaning data...")

df = df_raw.copy()

# TotalCharges contains 11 whitespace-only strings → treat as NaN → impute
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(
    f"[INFO] TotalCharges: {n_missing} whitespace rows coerced to NaN → imputed with median"
)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop customerID — not a predictive feature
df = df.drop(columns=["customerID"])

print(f"[RESULT] Nulls remaining after cleaning: {df.isnull().sum().sum()}")

# %%
# 3. ENCODE
# Binary Yes/No columns → 0/1
# Multi-class categoricals → LabelEncoder
# Trees don't require scaling, but they do require numeric input.

print("\n[STEP 3] Encoding categorical columns...")

for col in BINARY_COLS:
    df[col] = (df[col] == "Yes").astype(int)

# Yes/No/No service → 1/0/0 (treat "No X service" as a form of No)
for col in YES_NO_COLS:
    df[col] = (df[col] == "Yes").astype(int)

# Multi-class → integer labels
le = LabelEncoder()
for col in MULTI_COLS:
    df[col] = np.array(le.fit_transform(df[col]), dtype=int)

# Target
df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

print(
    f"[RESULT] Encoding complete. Churn balance:\n{df[TARGET_COL].value_counts().to_string()}"
)
print(f"Churn rate: {df[TARGET_COL].mean() * 100:.1f}%")

# %%
# 4. FEATURES / TARGET SPLIT

FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\n[STEP 4] Feature matrix : {X.shape}")
print(f"Feature names: {FEATURE_COLS}")

# %%
# 5. DECISION TREE — TRAIN

print(f"\n[STEP 5] Training Decision Tree (max_depth={DT_MAX_DEPTH})...")

dt = DecisionTreeClassifier(
    max_depth=DT_MAX_DEPTH,
    criterion="gini",
    random_state=RANDOM_STATE,
)
dt.fit(X, y)
print("[RESULT] Decision Tree trained.")

# %%
# 6. VISUALIZE TREE

print("\n[STEP 6] Visualizing tree...")

fig, ax = plt.subplots(figsize=(26, 10))
plot_tree(
    dt,
    feature_names=FEATURE_COLS,
    class_names=["No Churn", "Churned"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title(
    f"Decision Tree — Telco Customer Churn (max_depth={DT_MAX_DEPTH})",
    fontsize=14,
    pad=12,
)
plt.tight_layout()
tree_viz_path = os.path.join(OUTPUTS_DIR, "decision_tree_structure.png")
plt.savefig(tree_viz_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[RESULT] Tree visualization saved → {tree_viz_path}")

# Text form (useful for logging / reading root logic)
print("\n[INFO] Decision Tree — text representation:")
print(export_text(dt, feature_names=FEATURE_COLS))

# %%
# 7. WHY TOP-OF-TREE FEATURES MATTER
# The ROOT node is the single feature that splits the entire dataset most
# cleanly (highest Gini gain). Features lower in the tree only refine
# decisions for progressively smaller subsets — they matter less globally.
# Inspecting the top 2–3 levels of the tree tells you almost the whole story.

print("[STEP 7] Decision Tree — feature importances (ranked):")
dt_importances = pd.Series(dt.feature_importances_, index=FEATURE_COLS).sort_values(
    ascending=False
)
print(dt_importances[dt_importances > 0].to_string())

top_feature = dt_importances.idxmax()
print(f"""
[INSIGHT] Root split feature: '{top_feature}'
  → This is the single variable that divides churn vs no-churn most
    cleanly across ALL 7,043 customers.
  → Features near the top of the tree carry global signal.
  → Features at the leaves only affect small corner cases.
""")

# %%
# 8. RANDOM FOREST — TRAIN

print(f"[STEP 8] Training Random Forest ({RF_N_ESTIMATORS} trees)...")

rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X, y)
print("[RESULT] Random Forest trained.")

# %%
# 9. CROSS-VALIDATION COMPARISON

print(f"\n[STEP 9] Running {CV_N_SPLITS}-fold Stratified CV...")

cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

dt_cv_scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")
rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

print(
    f"\n[RESULT] Decision Tree  — Mean: {dt_cv_scores.mean():.4f}  Std: {dt_cv_scores.std():.4f}"
)
print(f"Scores per fold: {np.round(dt_cv_scores, 4)}")
print(
    f"\n[RESULT] Random Forest  — Mean: {rf_cv_scores.mean():.4f}  Std: {rf_cv_scores.std():.4f}"
)
print(f"Scores per fold: {np.round(rf_cv_scores, 4)}")
print(
    f"\n[RESULT] RF improvement over DT: +{(rf_cv_scores.mean() - dt_cv_scores.mean()) * 100:.2f}%"
)

# %%
# 10. FEATURE IMPORTANCE COMPARISON PLOT

rf_importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(
    ascending=False
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(
    dt_importances[dt_importances > 0].index,
    dt_importances[dt_importances > 0].values,
    color="steelblue",
    edgecolor="white",
)
axes[0].set_title("Decision Tree — Feature Importances", fontsize=12)
axes[0].set_xlabel("Gini Importance")
axes[0].invert_yaxis()

axes[1].barh(
    rf_importances.head(15).index,
    rf_importances.head(15).values,
    color="seagreen",
    edgecolor="white",
)
axes[1].set_title(
    f"Random Forest — Feature Importances (top 15, {RF_N_ESTIMATORS} trees)",
    fontsize=12,
)
axes[1].set_xlabel("Mean Decrease in Impurity")
axes[1].invert_yaxis()

plt.suptitle(
    "Feature Importance: Single Tree vs Forest — Telco Churn",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
fi_path = os.path.join(OUTPUTS_DIR, "dt_vs_rf_feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[RESULT] Feature importance comparison saved → {fi_path}")

# %%
# 11. CV SCORE COMPARISON PLOT

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(CV_N_SPLITS)
width = 0.35

ax.bar(
    x - width / 2,
    dt_cv_scores,
    width,
    label="Decision Tree",
    color="steelblue",
    alpha=0.85,
)
ax.bar(
    x + width / 2,
    rf_cv_scores,
    width,
    label="Random Forest",
    color="seagreen",
    alpha=0.85,
)
ax.axhline(
    dt_cv_scores.mean(), color="steelblue", linestyle="--", linewidth=1.2, alpha=0.7
)
ax.axhline(
    rf_cv_scores.mean(), color="seagreen", linestyle="--", linewidth=1.2, alpha=0.7
)

ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_title(
    f"{CV_N_SPLITS}-Fold Stratified CV: Decision Tree vs Random Forest — Telco Churn"
)
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {i + 1}" for i in range(CV_N_SPLITS)])
ax.set_ylim(0.70, 0.85)
ax.legend()
plt.tight_layout()
cv_path = os.path.join(OUTPUTS_DIR, "dt_vs_rf_cv_comparison.png")
plt.savefig(cv_path, dpi=150)
plt.close()
print(f"[RESULT] CV comparison plot saved → {cv_path}")

# %%
# SUMMARY

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DAY 10 — STANDARD TASK COMPLETE                                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Dataset : Telco Customer Churn  (7,043 rows × 19 features)              ║
║  Target  : Churn (26.5% positive rate)                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Decision Tree (depth {DT_MAX_DEPTH})                                                 ║
║    CV Accuracy  : {dt_cv_scores.mean():.4f}  ±  {dt_cv_scores.std():.4f}                                      ║
║    Top feature  : {top_feature:<30}                         ║
║    Interpretable, but high variance across folds                         ║
║                                                                          ║
║  Random Forest ({RF_N_ESTIMATORS} trees, depth {RF_MAX_DEPTH})                                      ║
║    CV Accuracy  : {rf_cv_scores.mean():.4f}  ±  {rf_cv_scores.std():.4f}                                      ║
║    Averages {RF_N_ESTIMATORS} trees → lower variance, stabler importances              ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Key Insight: Features at the ROOT of the tree carry the most            ║
║  global signal. RF importances are more reliable because they            ║
║  average impurity reduction across all {RF_N_ESTIMATORS} trees, not just one.          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
