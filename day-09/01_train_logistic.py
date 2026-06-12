# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, OUTPUTS_DIR, RANDOM_STATE, TARGET_COL, TEST_SIZE

# %%
df = pd.read_csv(DATA_PATH, na_values="?")
df = df.dropna()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL]).gt(0).astype(int)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# %% Scale - required for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only - no leakage

# %% Train
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AOC: {roc_auc:.4f}")

# %% Plot: 3-panel - ROC | Confusion MAtrix | Precision-Recall
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Logistic Regression - Diagnostic Plots", fontsize=14, fontweight="bold")

# Panel 1: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend(loc="lower right")

# Panel 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=["Pred: No Disease", "Pred: Disease"],
    yticklabels=["True: No Disease", "True: Disease"],
)
axes[1].set_title("Confusion Matrix (threshold = 0.5)")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# %% Panel 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
axes[2].plot(
    recall, precision, color="darkorange", lw=2, label=f"PR-AUC = {pr_auc:.3f}"
)
axes[2].axhline(
    y=y_test.mean(), color="gray", linestyle="--", alpha=0.7, label="Baseline (random)"
)
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-Recall Curve")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "01_diagnostic_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_diagnostic_plots.png")
