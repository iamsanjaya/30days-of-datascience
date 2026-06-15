# %% Imports
# Out-of-Box Challenge: "The Metric You Pick Changes Everything"
#
# The same XGBoost model, the same probabilities — but the threshold we cut at
# determines what the "best" model looks like.
#
# A model is not just its weights. It is also its operating point.
# This script makes that concrete.

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import OUTPUTS, load_and_preprocess, make_skf, make_xgb_model

warnings.filterwarnings("ignore")

# %% Load data & train model
X, y = load_and_preprocess()

# Use the last fold as the held-out test set — same split as script 01
skf = make_skf()
train_idx, test_idx = list(skf.split(X, y))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model = make_xgb_model()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_proba = model.predict_proba(X_test)[:, 1]

# %% Part 1: Metric landscape across all thresholds
# For every threshold from 0 to 1, compute all four metrics.
# This reveals: each metric peaks at a *different* threshold.

thresholds = np.linspace(0.01, 0.99, 200)
metrics_over_thresholds = {"Accuracy": [], "F1": [], "Precision": [], "Recall": []}

for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    metrics_over_thresholds["Accuracy"].append(accuracy_score(y_test, y_pred_t))
    metrics_over_thresholds["F1"].append(f1_score(y_test, y_pred_t, zero_division=0))
    metrics_over_thresholds["Precision"].append(
        precision_score(y_test, y_pred_t, zero_division=0)
    )
    metrics_over_thresholds["Recall"].append(
        recall_score(y_test, y_pred_t, zero_division=0)
    )

# Find optimal threshold per metric
optimal_thresholds = {}
for metric_name, scores in metrics_over_thresholds.items():
    best_idx = np.argmax(scores)
    optimal_thresholds[metric_name] = {
        "threshold": thresholds[best_idx],
        "score": scores[best_idx],
    }

print("Optimal threshold per metric")
for metric_name, info in optimal_thresholds.items():
    print(
        f"{metric_name:<12} → threshold = {info['threshold']:.3f}  |  score = {info['score']:.4f}"
    )

# %% Plot 1: All metrics across threshold range
COLOURS = {
    "Accuracy": "#2D6A9F",
    "F1": "#27AE60",
    "Precision": "#E74C3C",
    "Recall": "#E07B3B",
}

fig, ax = plt.subplots(figsize=(10, 6))
for metric_name, scores in metrics_over_thresholds.items():
    ax.plot(thresholds, scores, label=metric_name, color=COLOURS[metric_name], lw=2)
    opt = optimal_thresholds[metric_name]
    ax.axvline(
        opt["threshold"], color=COLOURS[metric_name], linestyle=":", alpha=0.5, lw=1
    )
    ax.scatter(
        [opt["threshold"]], [opt["score"]], color=COLOURS[metric_name], zorder=5, s=60
    )

ax.axvline(0.5, color="black", linestyle="--", lw=1, label="Default threshold (0.50)")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Metric Score")
ax.set_title(
    "Each Metric Peaks at a Different Threshold\n"
    "The 'Best Model' Depends Entirely on What You Optimise"
)
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(OUTPUTS / "threshold_analysis.png", dpi=150)
plt.close()
print("\nSaved: threshold_analysis.png")

# %% Plot 2: Confusion matrix at each optimal threshold
# Key takeaway: the *same probabilities* produce very different operational
# decisions depending on which metric we optimise for.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (metric_name, info) in enumerate(optimal_thresholds.items()):
    t = info["threshold"]
    y_pred_t = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    tn, fp, fn, tp = cm.ravel()

    ax = axes[i]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Predicted: Died", "Predicted: Survived"],
        yticklabels=["Actual: Died", "Actual: Survived"],
        cbar=False,
    )
    ax.set_title(
        f"Optimised for {metric_name}\n"
        f"Threshold = {t:.3f}  |  {metric_name} = {info['score']:.4f}\n"
        f"FN (missed survivors) = {fn}  |  FP (false alarms) = {fp}",
        fontsize=10,
    )

plt.suptitle(
    "Same Model, Same Probabilities — Four Different Operating Decisions\n"
    "False Negatives (FN) = Survivors the Model Missed",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
plt.savefig(OUTPUTS / "metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: metric_comparison.png")

# %% Part 3: Business/ethics analysis — Titanic rescue scenario
print("\nOut-of-Box Challenge: Choose Your Threshold Like a Rescuer")
print("""
Framing:
    - Positive class = Survived
    - False Negative (FN) = Model said "died," but person was alive → LEFT BEHIND
    - False Positive (FP) = Model said "survived," but person was dead → Wasted rescue resource

The ethical argument for maximising Recall in real-time rescue:
    - The cost of a FN is a human life
    - The cost of a FP is a wasted lifeboat trip
    - These are NOT equivalent costs
    - Therefore: we accept lower Precision to drive FN → 0

Domain-driven threshold recommendation: use Recall-optimised threshold.
""")

recall_opt = optimal_thresholds["Recall"]
t_recall = recall_opt["threshold"]
y_pred_recall = (y_proba >= t_recall).astype(int)
cm_recall = confusion_matrix(y_test, y_pred_recall)
tn, fp, fn, tp = cm_recall.ravel()

y_pred_default = (y_proba >= 0.5).astype(int)
cm_default = confusion_matrix(y_test, y_pred_default)

acc_default = accuracy_score(y_test, y_pred_default)
acc_recall = accuracy_score(y_test, y_pred_recall)

print(f"Default threshold (0.50) → Accuracy: {acc_default:.4f}")
print(
    f"Recall threshold  ({t_recall:.3f}) → Accuracy: {acc_recall:.4f}  (intentionally lower)"
)
print(f"Recall: {recall_opt['score']:.4f}  (we catch more survivors)")
print(f"FN = {fn}  (survivors missed — the cost we are minimising)")
print(f"FP = {fp}  (false alarms — acceptable in rescue context)")
print(
    f"\nConclusion: a {acc_default - acc_recall:.2%} drop in accuracy is a fair trade "
    f"to recover {cm_recall[1, 1]} vs {cm_default[1, 1]} survivors."
)

# %% Final comparison table
print("\nSummary: What Each Metric Optimisation Produces")
rows = []
for metric_name, info in optimal_thresholds.items():
    t = info["threshold"]
    y_pred_t = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    rows.append(
        {
            "Optimised For": metric_name,
            "Threshold": round(t, 3),
            "Accuracy": round(accuracy_score(y_test, y_pred_t), 4),
            "F1": round(float(f1_score(y_test, y_pred_t, zero_division=0)), 4),
            "Precision": round(
                float(precision_score(y_test, y_pred_t, zero_division=0)), 4
            ),
            "Recall": round(float(recall_score(y_test, y_pred_t, zero_division=0)), 4),
            "FN (missed)": fn,
            "FP (false alarm)": fp,
        }
    )

summary_df = pd.DataFrame(rows).set_index("Optimised For")
print(summary_df.to_string())
