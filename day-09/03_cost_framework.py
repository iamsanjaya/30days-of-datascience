# %% Imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

# %% Setup
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

# %% Cost function
# Total cost = (# FN * FN_cost) + (# FP * FP_cost)
# The threshold that minimizes total cost is the optimal medical threshold.
# This is NOT the same as the threshold that maximizes F1.

thresholds = np.linspace(0.01, 0.99, 200)
costs: list[float] = []
fn_list: list[int] = []
fp_list: list[int] = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    total_cost = (fn * FALSE_NEGATIVE_COST) + (fp * FALSE_POSITIVE_COST)
    costs.append(float(total_cost))
    fn_list.append(int(fn))
    fp_list.append(int(fp))

optimal_idx = int(np.argmin(costs))
optimal_threshold = float(thresholds[optimal_idx])
min_cost = float(costs[optimal_idx])

# Cost at default threshold for comparison
default_idx = int(np.argmin(np.abs(thresholds - 0.5)))
default_cost = float(costs[default_idx])

print(
    f"FN cost weight: {FALSE_NEGATIVE_COST}x  |  FP cost weight: {FALSE_POSITIVE_COST}x"
)
print(
    f"Optimal threshold (cost-based): {optimal_threshold:.2f}  (total cost: {min_cost:.0f})"
)
print(f"Default threshold (0.5):        0.50  (total cost: {default_cost:.0f})")
print(f"Cost reduction by shifting threshold: {default_cost - min_cost:.0f} units")

# %% Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Cost-Based Threshold Selection — FN costs {FALSE_NEGATIVE_COST}× more than FP",
    fontsize=13,
    fontweight="bold",
)

# Panel 1: Total cost curve
ax = axes[0]
ax.plot(thresholds, costs, color="crimson", lw=2, label="Total cost")
ax.axvline(0.5, color="gray", linestyle=":", lw=1.5, label="Default (0.5)")
ax.axvline(
    optimal_threshold,
    color="seagreen",
    linestyle="--",
    lw=2,
    label=f"Optimal ({optimal_threshold:.2f})",
)
ax.scatter([optimal_threshold], [min_cost], color="seagreen", zorder=5, s=80)
ax.scatter([thresholds[default_idx]], [default_cost], color="gray", zorder=5, s=80)
ax.annotate(
    f"Cost saved:\n{default_cost - min_cost:.0f} units",
    xy=(optimal_threshold, min_cost),
    xytext=(optimal_threshold + 0.12, min_cost + 5),
    arrowprops={"arrowstyle": "->", "color": "black"},
    fontsize=9,
)
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Total Cost (weighted errors)")
ax.set_title("Minimizing Harm: Not All Errors Are Equal")
ax.legend()

# Panel 2: FN vs FP tradeoff
ax2 = axes[1]
ax2.plot(
    thresholds,
    fn_list,
    color="tomato",
    lw=2,
    label=f"False Negatives (cost ×{FALSE_NEGATIVE_COST})",
)
ax2.plot(
    thresholds,
    fp_list,
    color="steelblue",
    lw=2,
    label=f"False Positives (cost ×{FALSE_POSITIVE_COST})",
)
ax2.axvline(0.5, color="gray", linestyle=":", lw=1.5, alpha=0.8)
ax2.axvline(optimal_threshold, color="seagreen", linestyle="--", lw=2, alpha=0.9)
ax2.set_xlabel("Decision Threshold")
ax2.set_ylabel("Count")
ax2.set_title("The Tradeoff: Lowering Threshold Trades FN for FP")
ax2.legend()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "03_cost_framework.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_cost_framework.png")
