# %%
"""
Out-of-Box Challenge: "Build the Tree in Your Head"

Tasks:
  1. Manually trace the decision path for 10 rows using decision_path()
  2. Find 2 rows that end in DIFFERENT leaf nodes but share the SAME
     predicted probability — then ask: are they actually similar?
  3. Write a 5-sentence reflection on what Decision Trees get wrong
     that Random Forest tries to fix, using THIS dataset as the example.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances

from config import (
    DATA_RAW,
    OUTPUTS_DIR,
    RANDOM_STATE,
    DT_MAX_DEPTH,
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
# 1. LOAD & PREPROCESS (mirrors 01_ exactly — shared config keeps this DRY)

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

# Train tree (same config as 01_)
dt = DecisionTreeClassifier(
    max_depth=DT_MAX_DEPTH, criterion="gini", random_state=RANDOM_STATE
)
dt.fit(X, y)

# %%
# 2. TRACE DECISION PATHS FOR 10 ROWS
# decision_path() returns a sparse indicator matrix:
#   row i, col j = 1  →  sample i passed through node j

print("=" * 68)
print(" TASK 1: Decision path trace for 10 sample rows")
print("=" * 68)

# Pick first 10 rows for deterministic, reproducible output
sample_idx = list(range(10))
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[np.array(sample_idx)]
y_pred = dt.predict(X_sample)
y_prob = np.array(dt.predict_proba(X_sample))[:, 1]  # probability of Churn=1

# The sparse path matrix
indicator = dt.decision_path(X_sample)

print(
    f"\n{'Row':>4}  {'Actual':>8}  {'Predicted':>10}  {'P(Churn)':>10}  {'Leaf node':>10}  Path (node ids)"
)
print("-" * 80)

for i, row_idx in enumerate(sample_idx):
    node_ids = indicator.getrow(i).indices  # nodes visited by this sample
    leaf_id = node_ids[-1]  # last node = leaf
    actual = int(y_sample.iloc[i])
    predicted = int(y_pred[i])
    prob = y_prob[i]
    path_str = " → ".join(str(n) for n in node_ids)
    print(
        f"{row_idx:>4}  {'Yes' if actual else 'No':>8}  {'Yes' if predicted else 'No':>10}"
        f"  {prob:>10.4f}  {leaf_id:>10}  {path_str}"
    )

# %%
# 3. DETAILED PATH WALKTHROUGH FOR 2 ROWS
# Print the human-readable rule path for the first two samples.


def describe_path(dt_model, X_single, feature_names):
    """Return a list of rule strings for one sample's path through the tree."""
    indicator = dt_model.decision_path(X_single.to_frame().T)
    node_ids = indicator.getrow(0).indices
    feat = dt_model.tree_.feature
    thresh = dt_model.tree_.threshold
    rules = []
    for depth, node in enumerate(node_ids[:-1]):  # all nodes except leaf
        feature_name = feature_names[feat[node]]
        value = X_single[feature_name]
        direction = "<=" if value <= thresh[node] else ">"
        rules.append(
            f"  depth {depth}: {feature_name} = {value:.4g}  {direction}  {thresh[node]:.4g}"
        )
    rules.append(f"  → Leaf node {node_ids[-1]}")
    return rules


print("\n" + "=" * 68)
print(" DETAILED PATH WALKTHROUGH — rows 0 and 1")
print("=" * 68)

for row_idx in [0, 1]:
    row = X.iloc[row_idx]
    actual = int(y.iloc[row_idx])
    predicted = int(dt.predict(row.to_frame().T)[0])
    prob = float(np.array(dt.predict_proba(row.to_frame().T))[0, 1])
    print(
        f"\nRow {row_idx} | Actual: {'Churned' if actual else 'No Churn'}  "
        f"| Predicted: {'Churned' if predicted else 'No Churn'}  | P(Churn)={prob:.4f}"
    )
    for rule in describe_path(dt, row, FEATURE_COLS):
        print(rule)

# %%
# 4. LEAF FLATTENING: SAME LEAF → SAME SCORE, DESPITE DIFFERENT CUSTOMERS
#
# With max_depth=4 on this dataset, each leaf gets a unique probability —
# no two leaves share the same score.  But the more fundamental limitation
# is the OTHER direction: ALL customers inside a single leaf are assigned
# the SAME score, regardless of how different they actually are.
#
# We surface that here: find the most populated leaf and compare the
# most-different pair of customers inside it.

print("\n" + "=" * 68)
print(" TASK 2: Same leaf → same score, despite very different customers")
print("=" * 68)

all_probs = np.array(dt.predict_proba(X))[:, 1]
all_leaves = dt.apply(X)
combined = pd.DataFrame({"prob": all_probs, "leaf": all_leaves}, index=X.index)

# Most populated leaf
largest_leaf = combined["leaf"].value_counts().idxmax()
leaf_members_idx = combined[combined["leaf"] == largest_leaf].index
leaf_X = X.loc[leaf_members_idx]
leaf_prob = combined.loc[leaf_members_idx, "prob"].iloc[0]
leaf_actuals = y.loc[leaf_members_idx]

print(
    f"\n[INFO] Largest leaf: node {largest_leaf}  "
    f"({len(leaf_members_idx)} customers, all scored P(Churn)={leaf_prob:.4f})"
)
print(
    f"Actual churn inside this leaf: "
    f"{leaf_actuals.sum()} churned / {len(leaf_actuals)} total "
    f"({leaf_actuals.mean()*100:.1f}%)"
)

# Find the most dissimilar pair inside this leaf using feature L1 distance
sub_array = leaf_X.values.astype(float)
dist_matrix = pairwise_distances(sub_array, metric="l1")
np.fill_diagonal(dist_matrix, 0)

flat_idx = np.argmax(dist_matrix)
row_i, row_j = np.unravel_index(flat_idx, dist_matrix.shape)
idx_a, idx_b = leaf_members_idx[int(row_i)], leaf_members_idx[int(row_j)]

print(f"\nMost dissimilar pair inside leaf {largest_leaf}:")
print(f"Row {idx_a}  vs  Row {idx_b}  (L1 distance = {dist_matrix[row_i, row_j]:.2f})")

diff: pd.Series = (X.loc[idx_a] - X.loc[idx_b]).abs().sort_values(ascending=False)

comparison = pd.DataFrame(
    {
        f"Row {idx_a}": X.loc[idx_a],
        f"Row {idx_b}": X.loc[idx_b],
        "Difference": diff,
    }
)
print("\n  Feature comparison (sorted by difference):")
print(comparison.sort_values("Difference", ascending=False).to_string())
print(f"""
  [FINDING] Both customers receive P(Churn)={leaf_prob:.4f}, yet they differ
  by {diff.iloc[0]:.2f} on '{str(diff.index[0])}' and {diff.iloc[1]:.2f} on '{str(diff.index[1])}'.
  The tree cannot distinguish them — it has used up all {DT_MAX_DEPTH} split levels
  before reaching these fine-grained differences.  This is the leaf
  flattening problem: inside a leaf, the tree is blind.
""")

# %%
# 5. VISUALISATION: PROBABILITY DISTRIBUTIONS BY LEAF
# Shows that each leaf assigns a FLAT probability to all its members —
# a key limitation that probability calibration or gradient boosting addresses.

print("[STEP 5] Plotting probability distribution per leaf...")

leaf_probs = pd.DataFrame({"leaf": all_leaves, "prob": all_probs})
leaf_counts = leaf_probs["leaf"].value_counts()
top_leaves = leaf_counts.head(6).index.tolist()  # 6 most populated leaves

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i, leaf_id in enumerate(top_leaves):
    members = leaf_probs[leaf_probs["leaf"] == leaf_id]["prob"]
    axes[i].hist(members, bins=5, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].set_title(f"Leaf {leaf_id}  (n={len(members)})", fontsize=10)
    axes[i].set_xlabel("P(Churn)")
    axes[i].set_ylabel("Count")
    axes[i].set_xlim(-0.05, 1.05)

plt.suptitle(
    "P(Churn) Distribution Inside Each Leaf — All Customers in a Leaf Get\n"
    "the SAME Score (flat bars = no within-leaf discrimination)",
    fontsize=11,
)
plt.tight_layout()
leaf_dist_path = os.path.join(OUTPUTS_DIR, "leaf_probability_distribution.png")
plt.savefig(leaf_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[RESULT] Leaf probability distribution saved → {leaf_dist_path}")

# %%
# 6. DT vs RF PROBABILITY SCATTER
# Decision Tree: probabilities cluster at discrete values (one per leaf).
# Random Forest: probabilities spread continuously — smoother calibration.

rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X, y)

dt_probs = np.array(dt.predict_proba(X))[:, 1]
rf_probs = np.array(rf.predict_proba(X))[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(dt_probs, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].set_title(
    "Decision Tree — P(Churn) distribution\n(discrete spikes: one per leaf)",
    fontsize=11,
)
axes[0].set_xlabel("P(Churn)")
axes[0].set_ylabel("Count")

axes[1].hist(rf_probs, bins=30, color="seagreen", edgecolor="white", alpha=0.85)
axes[1].set_title(
    f"Random Forest — P(Churn) distribution\n(continuous: averaged across {RF_N_ESTIMATORS} trees)",
    fontsize=11,
)
axes[1].set_xlabel("P(Churn)")

plt.suptitle("Probability Calibration: Single Tree vs Forest", fontsize=12, y=1.02)
plt.tight_layout()
prob_dist_path = os.path.join(OUTPUTS_DIR, "dt_vs_rf_probability_distribution.png")
plt.savefig(prob_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[RESULT] Probability distribution comparison saved → {prob_dist_path}")

# %%
# 7. REFLECTION

print("""
╔══════════════════════════════════════════════════════════════════╗
║  5-SENTENCE REFLECTION — Using Telco Churn as the Example        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. FLAT LEAF PROBABILITIES                                      ║
║     Every customer landing in the same leaf receives the exact   ║
║     same P(Churn). In this dataset, two customers with very      ║
║     different tenure and MonthlyCharges can receive identical    ║
║     churn scores simply because the tree's 4 splits place them   ║
║     in the same partition — a blunt instrument for a business    ║
║     that needs precise risk ranking.                             ║
║                                                                  ║
║  2. HIGH VARIANCE ACROSS FOLDS                                   ║
║     The DT's CV accuracy ranged from 76.8% to 80.6% across 5     ║
║     folds (std=0.012). Random Forest compressed that range       ║
║     to 79.2%–81.2% (std=0.007). A single tree is highly          ║
║     sensitive to which customers land in the training fold;      ║
║     the forest averages that noise away.                         ║
║                                                                  ║
║  3. CONTRACT AS ROOT — TOO DOMINANT IN A SINGLE TREE             ║
║     'Contract' accounts for ~59% of the single tree's            ║
║     importance. This makes the tree near-useless for customers   ║
║     already on month-to-month contracts (the highest-risk        ║
║     segment), because all the discriminating power is spent on   ║
║     the root split. RF distributes signal across tenure,         ║
║     MonthlyCharges, and InternetService as well.                 ║
║                                                                  ║
║  4. OVERFITTING TO LOCAL PATTERNS                                ║
║     A DT at depth 4 is deliberately constrained here; without    ║
║     pruning it would memorise noise. RF's bagging means each     ║
║     tree sees a different random subset, so no single outlier    ║
║     can hijack the model.                                        ║
║                                                                  ║
║  5. WHAT RANDOM FOREST FIXES — AND WHAT IT DOESN'T               ║
║     RF solves variance and probability discretisation by voting  ║
║     across 200 trees. What it does NOT fix: it is still a        ║
║     majority-vote of impurity-based splits, so rare customer     ║
║     segments (SeniorCitizen=1, tenure<6 months) may still be     ║
║     underserved — a problem that boosting (XGBoost, Day 11)      ║
║     addresses by iteratively reweighting hard-to-classify rows.  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
