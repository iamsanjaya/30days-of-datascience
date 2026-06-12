# %% Imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL, TEST_SIZE

# %% Setup (same train/test as script 01)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# %% Threshold sweep
thresholds = np.linspace(0.01, 0.99, 200)
metrics: dict[str, list[float]] = {
    "precision": [],
    "recall": [],
    "f1": [],
    "specificity": [],
}

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    metrics["precision"].append(
        float(precision_score(y_test, y_pred_t, zero_division=0))
    )
    metrics["recall"].append(float(recall_score(y_test, y_pred_t, zero_division=0)))
    metrics["f1"].append(float(f1_score(y_test, y_pred_t, zero_division=0)))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    metrics["specificity"].append(float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0)

# %% Plot: threshold sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("How Threshold Choice Changes Everything", fontsize=13, fontweight="bold")

# Panel 1: Precision, Recall, F1 vs Threshold
ax = axes[0]
ax.plot(thresholds, metrics["precision"], label="Precision", color="steelblue", lw=2)
ax.plot(
    thresholds, metrics["recall"], label="Recall (Sensitivity)", color="tomato", lw=2
)
ax.plot(
    thresholds, metrics["f1"], label="F1 Score", color="seagreen", lw=2, linestyle="--"
)
ax.axvline(0.5, color="gray", linestyle=":", alpha=0.8, label="Default (0.5)")

best_f1_idx = int(np.argmax(metrics["f1"]))
best_t = float(thresholds[best_f1_idx])
ax.axvline(
    best_t, color="seagreen", linestyle="--", alpha=0.6, label=f"Best F1 @ {best_t:.2f}"
)

ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision vs Recall vs F1 — Every Threshold Tells a Different Story")
ax.legend()
ax.set_ylim(0, 1.05)

# Panel 2: What the confusion matrix looks like at 3 key thresholds
key_thresholds = [0.2, 0.5, 0.8]
colors = ["tomato", "gray", "steelblue"]
labels = ["0.2 (catch more disease)", "0.5 (default)", "0.8 (very conservative)"]

recall_vals = []
precision_vals = []
fn_counts = []
fp_counts = []

for t in key_thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    recall_vals.append(tp / (tp + fn))
    precision_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    fn_counts.append(fn)
    fp_counts.append(fp)

x = np.arange(len(key_thresholds))
width = 0.35

ax2 = axes[1]
bars_fn = ax2.bar(
    x - width / 2,
    fn_counts,
    width,
    label="False Negatives (missed disease)",
    color="tomato",
    alpha=0.85,
)
bars_fp = ax2.bar(
    x + width / 2,
    fp_counts,
    width,
    label="False Positives (false alarm)",
    color="steelblue",
    alpha=0.85,
)

ax2.bar_label(bars_fn, padding=2)
ax2.bar_label(bars_fp, padding=2)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Count (out of test set)")
ax2.set_title("Error Tradeoff: Lowering Threshold Reduces Missed Cases")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "02_threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print("Threshold analysis results:")
for i, t in enumerate(key_thresholds):
    print(
        f"  t={t}: recall={recall_vals[i]:.3f}, precision={precision_vals[i]:.3f}, "
        f"FN={fn_counts[i]}, FP={fp_counts[i]}"
    )
print(f"\nBest F1 threshold: {best_t:.2f}")
print("Saved: 02_threshold_analysis.png")
