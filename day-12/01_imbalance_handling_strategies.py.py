# %%
# Standard Task — Class Imbalance Strategy Comparison
# Dataset  : Breast Cancer (Status: Dead = minority, Alive = majority)
# Strategies: SMOTE · Undersampling · class_weight='balanced' · Baseline
# Metrics  : F1 (macro), ROC-AUC, Precision, Recall
# Output   : outputs/strategy_comparison.png


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import os
import warnings

from config import DATA_PATH, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL

warnings.filterwarnings("ignore")

# Paths
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# %% 1. Load & encode
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape : {df.shape}")
print(f"Target distribution:\n{df[TARGET_COL].value_counts()}\n")

# Drop rows where target is missing
df = df.dropna(subset=[TARGET_COL])

# Encode target  →  Dead = 1 (minority / positive class), Alive = 0
le_target = LabelEncoder()
y = np.asarray(
    le_target.fit_transform(df[TARGET_COL]), dtype=np.int64
)  # Dead=1, Alive=0
encoded_classes = np.asarray(le_target.transform(le_target.classes_))
print("Classes after encoding:" f"{list(zip(le_target.classes_, encoded_classes))}")
print(f"Positive (Dead) ratio: {y.mean():.3f}\n")

# Drop target + any ID-like columns
X = df.drop(columns=[TARGET_COL])

# Encode remaining categoricals
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = pd.Series(le.fit_transform(X[col].astype(str)), index=X.index)

X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Feature matrix shape : {X_scaled.shape}")
print(f"Class counts  → 0 (Alive): {(y == 0).sum()}  | 1 (Dead): {(y == 1).sum()}\n")


# %% 2. CV evaluation helper
def evaluate_strategy(X_arr, y_arr, strategy_name, resample_fn=None):
    """
    Runs 5-fold Stratified CV.  Resampling (if any) is applied inside each
    fold on the training split only — no leakage.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"f1": [], "auc": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        # Apply resampling only to training fold
        if resample_fn is not None:
            X_tr, y_tr = resample_fn(X_tr, y_tr)

        # Use class_weight for the balanced strategy
        cw = "balanced" if strategy_name == "class_weight_balanced" else None
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight=cw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        fold_metrics["f1"].append(
            f1_score(y_val, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["auc"].append(roc_auc_score(y_val, y_prob))
        fold_metrics["precision"].append(
            precision_score(y_val, y_pred, average="macro", zero_division=0)
        )
        fold_metrics["recall"].append(
            recall_score(y_val, y_pred, average="macro", zero_division=0)
        )

    means = {k: np.mean(v) for k, v in fold_metrics.items()}
    stds = {k: np.std(v) for k, v in fold_metrics.items()}

    print(
        f"[{strategy_name:25s}]  "
        f"F1={means['f1']:.3f}±{stds['f1']:.3f}  "
        f"AUC={means['auc']:.3f}±{stds['auc']:.3f}  "
        f"P={means['precision']:.3f}  R={means['recall']:.3f}"
    )
    return means, stds


# %% 3. Define strategies
X_np = X_scaled.values

smote = SMOTE(random_state=RANDOM_STATE)
undersampler = RandomUnderSampler(random_state=RANDOM_STATE)

strategies = {
    "baseline": None,
    "SMOTE": smote.fit_resample,
    "undersampling": undersampler.fit_resample,
    "class_weight_balanced": None,  # handled inside evaluate_strategy
}

# %% 4. Run all strategies
print("=" * 75)
print("Strategy Comparison — 5-Fold Stratified CV")
print("=" * 75)

results = {}
for name, fn in strategies.items():
    means, stds = evaluate_strategy(X_np, y, name, resample_fn=fn)
    results[name] = {"means": means, "stds": stds}

print("=" * 75)

# %% 5. Plot comparison
metrics = ["f1", "auc", "precision", "recall"]
metric_labels = ["F1 (macro)", "ROC-AUC", "Precision (macro)", "Recall (macro)"]
strategy_names = list(results.keys())
colors = ["#7f8c8d", "#2ecc71", "#e67e22", "#3498db"]

fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
fig.suptitle(
    "Day 12 — Class Imbalance: Strategy Comparison\n"
    "Breast Cancer Dataset | 5-Fold Stratified CV | Random Forest",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

for ax, metric, label in zip(axes, metrics, metric_labels):
    means_vals = [results[s]["means"][metric] for s in strategy_names]
    stds_vals = [results[s]["stds"][metric] for s in strategy_names]

    bars = ax.bar(
        range(len(strategy_names)),
        means_vals,
        yerr=stds_vals,
        color=colors,
        capsize=5,
        edgecolor="white",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"},
    )

    # Value labels on bars
    for bar, val in zip(bars, means_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in strategy_names],
        fontsize=8,
        rotation=15,
        ha="right",
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# Legend for strategy colours
handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors]
fig.legend(
    handles,
    strategy_names,
    loc="lower center",
    ncol=4,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

plt.tight_layout()
save_path = os.path.join(OUTPUTS_DIR, "strategy_comparison.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nOutput saved → {save_path}")

# %% 6. Best strategy summary
best_f1_strategy = max(results, key=lambda s: results[s]["means"]["f1"])
best_auc_strategy = max(results, key=lambda s: results[s]["means"]["auc"])

print("\nBest Strategies")
print(f"Best F1: {best_f1_strategy}  ({results[best_f1_strategy]['means']['f1']:.3f})")
print(
    f"Best AUC : {best_auc_strategy}  ({results[best_auc_strategy]['means']['auc']:.3f})"
)
print("\n")
