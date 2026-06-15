# %%
# Out-of-Box Challenge — Manufacture Your Own Imbalance
# Goal    : Systematically degrade class balance and observe metric collapse
# Ratios  : 1:2, 1:5, 1:10, 1:50, 1:100  (minority:majority)
# Metrics : Accuracy, F1 (macro), ROC-AUC, PR-AUC
# Outputs :
#   outputs/degradation_curves.png   ← metric vs imbalance ratio lines
#   outputs/metric_gap_heatmap.png   ← accuracy vs F1 gap reveals deception


import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder

from config import DATA_PATH, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL

warnings.filterwarnings("ignore")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %% 1. Load & prepare full balanced-ish dataset
df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET_COL])

le_target = LabelEncoder()
y_full = le_target.fit_transform(df[TARGET_COL])  # Dead=1 (minority)
y_full = le_target.fit_transform(df[TARGET_COL])
y_full = np.asarray(y_full, dtype=np.int64)

X_full = df.drop(columns=[TARGET_COL])
encoder = LabelEncoder()

for col in X_full.select_dtypes(include="object").columns:
    le = LabelEncoder()

    cat_cols = X_full.select_dtypes(include=["object", "string"]).columns

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    X_full[cat_cols] = encoder.fit_transform(X_full[cat_cols].astype(str))
    X_full = X_full.fillna(X_full.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
X_scaled = np.asarray(X_scaled, dtype=np.float64)

# Separate minority (Dead=1) and majority (Alive=0)
minority_idx = np.where(y_full == 1)[0]
majority_idx = np.where(y_full == 0)[0]

print(f"Total minority (Dead): {len(minority_idx)}")
print(f"Total majority (Alive): {len(majority_idx)}")
n_minority = len(minority_idx)

# %% 2. Imbalance simulation
# Each ratio = minority : majority  →  1:R means majority = R * n_minority
RATIOS = [2, 5, 10, 50, 100]  # 1:2  1:5  1:10  1:50  1:100


def build_imbalanced_dataset(ratio):
    """
    Subsample majority class to achieve minority:majority = 1:ratio.
    If we don't have enough majority samples, use all available.
    """
    n_majority_target = min(n_minority * ratio, len(majority_idx))
    rng = np.random.default_rng(RANDOM_STATE)

    maj_sample = rng.choice(majority_idx, size=n_majority_target, replace=False)

    idx = np.concatenate([minority_idx, maj_sample])
    idx = np.asarray(idx, dtype=np.int64)

    rng.shuffle(idx)

    return np.asarray(X_scaled)[idx], np.asarray(y_full)[idx]


def cv_metrics(X, y, use_smote=False):
    """5-fold CV, returns dict of mean metric scores."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    acc_list, f1_list, auc_list, prauc_list = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_smote and y_tr.sum() >= 2:
            try:
                sm = SMOTE(random_state=RANDOM_STATE)
                resampled = sm.fit_resample(X_tr, y_tr)
                X_tr = resampled[0]
                y_tr = resampled[1]
            except Exception:
                pass  # too few minority samples for SMOTE — skip

        clf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        acc_list.append(accuracy_score(y_val, y_pred))
        f1_list.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        auc_list.append(roc_auc_score(y_val, y_prob))
        prauc_list.append(average_precision_score(y_val, y_prob))

    return {
        "accuracy": np.mean(acc_list),
        "f1": np.mean(f1_list),
        "auc": np.mean(auc_list),
        "pr_auc": np.mean(prauc_list),
    }


# %% 3. Run experiment
print("\n" + "=" * 65)
print("Imbalance Degradation Study")
print("=" * 65)

baseline_metrics_no_smote = {}
baseline_metrics_smote = {}

for ratio in RATIOS:
    X_r, y_r = build_imbalanced_dataset(ratio)
    n_majority = np.sum(y_r == 0)
    n_minority = np.sum(y_r == 1)
    actual_ratio = n_majority / n_minority

    m_plain = cv_metrics(X_r, y_r, use_smote=False)
    m_smote = cv_metrics(X_r, y_r, use_smote=True)

    baseline_metrics_no_smote[ratio] = m_plain
    baseline_metrics_smote[ratio] = m_smote

    print(
        f"1:{ratio:3d}  (actual {actual_ratio:.1f}:1)  "
        f"| No SMOTE  Acc={m_plain['accuracy']:.3f}  F1={m_plain['f1']:.3f}  "
        f"AUC={m_plain['auc']:.3f}  PR-AUC={m_plain['pr_auc']:.3f}"
    )
    print(
        f"{'':>20}  "
        f"| SMOTE Acc={m_smote['accuracy']:.3f}  F1={m_smote['f1']:.3f}  "
        f"AUC={m_smote['auc']:.3f}  PR-AUC={m_smote['pr_auc']:.3f}"
    )
    print()

print("=" * 65)

# %% 4. Plot 1 — Degradation curves
metrics_to_plot = ["accuracy", "f1", "auc", "pr_auc"]
labels = ["Accuracy", "F1 (macro)", "ROC-AUC", "PR-AUC"]
colors_plain = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6"]
colors_smote = ["#f1948a", "#f0b27a", "#85c1e9", "#d2b4de"]

ratio_labels = [f"1:{r}" for r in RATIOS]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Day 12 Out-of-Box — Metric Degradation Under Manufactured Imbalance\n"
    "Breast Cancer Dataset | Solid = No SMOTE | Dashed = With SMOTE",
    fontsize=13,
    fontweight="bold",
)

for ax, metric, label, c_plain, c_smote in zip(
    axes.flat, metrics_to_plot, labels, colors_plain, colors_smote
):
    vals_plain = [baseline_metrics_no_smote[r][metric] for r in RATIOS]
    vals_smote = [baseline_metrics_smote[r][metric] for r in RATIOS]

    ax.plot(
        ratio_labels,
        vals_plain,
        marker="o",
        linewidth=2.2,
        color=c_plain,
        label="No SMOTE",
    )
    ax.plot(
        ratio_labels,
        vals_smote,
        marker="s",
        linewidth=2.2,
        color=c_smote,
        linestyle="--",
        label="SMOTE",
    )

    # Annotate values
    for i, (vp, vs) in enumerate(zip(vals_plain, vals_smote)):
        ax.annotate(
            f"{vp:.2f}",
            (i, vp),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7.5,
            color=c_plain,
        )
        ax.annotate(
            f"{vs:.2f}",
            (i, vs),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7.5,
            color="#7f8c8d",
        )

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Imbalance Ratio (minority:majority)", fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
save1 = os.path.join(OUTPUTS_DIR, "degradation_curves.png")
plt.savefig(save1, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved → {save1}")

# %% 5. Plot 2 — Accuracy vs F1 gap heatmap (the deception map)
# At high imbalance, accuracy stays high while F1 collapses.
# The gap (Acc - F1) reveals how deceptive accuracy is.

gap_no_smote = np.array(
    [
        baseline_metrics_no_smote[r]["accuracy"] - baseline_metrics_no_smote[r]["f1"]
        for r in RATIOS
    ]
)
gap_smote = np.array(
    [
        baseline_metrics_smote[r]["accuracy"] - baseline_metrics_smote[r]["f1"]
        for r in RATIOS
    ]
)

# Build a 2-row heatmap: row0 = No SMOTE, row1 = SMOTE
heatmap_data = np.vstack([gap_no_smote, gap_smote])

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
im = ax2.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

ax2.set_xticks(range(len(RATIOS)))
ax2.set_xticklabels(ratio_labels, fontsize=10)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["No SMOTE", "SMOTE"], fontsize=10)
ax2.set_xlabel("Imbalance Ratio (minority:majority)", fontsize=10)
ax2.set_title(
    "Accuracy − F1 Gap Heatmap\n"
    "Larger gap = Accuracy is more misleading as an imbalance metric",
    fontsize=11,
    fontweight="bold",
)

for row in range(2):
    for col in range(len(RATIOS)):
        val = heatmap_data[row, col]
        ax2.text(
            col,
            row,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white" if val > 0.25 else "#2c3e50",
        )

plt.colorbar(im, ax=ax2, label="Accuracy − F1 (macro)", shrink=0.8)
plt.tight_layout()
save2 = os.path.join(OUTPUTS_DIR, "metric_gap_heatmap.png")
plt.savefig(save2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {save2}")

# %% 6. Key findings
print("\nKey Findings ")
print(
    "► At 1:100 imbalance, Accuracy stays high while F1 collapses — "
    "accuracy is a useless metric here."
)

# Find ratio where F1 drops below 0.5 (model barely better than random per class)
for r in RATIOS:
    if baseline_metrics_no_smote[r]["f1"] < 0.5:
        print(f"► F1 drops below 0.50 (near-useless) at ratio 1:{r} without SMOTE.")
        break

for r in RATIOS:
    if baseline_metrics_smote[r]["f1"] < 0.5:
        print(f"► With SMOTE, F1 drops below 0.50 at ratio 1:{r}.")
        break

# Find ratio where SMOTE helps most
smote_gains = {
    r: baseline_metrics_smote[r]["f1"] - baseline_metrics_no_smote[r]["f1"]
    for r in RATIOS
}
best_ratio = max(smote_gains.items(), key=lambda x: x[1])[0]
print(
    f"► SMOTE provides the largest F1 gain (+{smote_gains[best_ratio]:.3f}) "
    f"at ratio 1:{best_ratio}."
)
print("\n")
