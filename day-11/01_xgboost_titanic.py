# %% Imports
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

from config import (
    OUTPUTS,
    N_FOLDS,
    RANDOM_STATE,
    load_and_preprocess,
    make_skf,
    make_xgb_model,
)

warnings.filterwarnings("ignore")

# %% Load data
X, y = load_and_preprocess()

print(f"Dataset shape: {X.shape}")
print(f"Survival rate: {y.mean():.2%}  (positive class imbalance check)")
print(f"Features: {X.columns.tolist()}\n")

# %% Stratified K-Fold cross-validation
# Stratified = each fold preserves the original 38/62 class ratio.
# Without it, fold-to-fold variance inflates and CV scores become unreliable.

skf = make_skf()

xgb_model = make_xgb_model()
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=RANDOM_STATE,
)

xgb_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring="roc_auc")
rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring="roc_auc")

print("Stratified K-Fold ROC-AUC (5 folds)")
print(f"XGBoost: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"fold scores: {np.round(xgb_scores, 4)}")
print(f"Rand Forest: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"fold scores: {np.round(rf_scores, 4)}")
delta = xgb_scores.mean() - rf_scores.mean()
print(
    f"\nXGBoost beats RF by: {delta:+.4f} AUC  ({'[OK] target met' if delta > 0.01 else '[--] target not met — consider tuning'})\n"
)

# %% Final model fit on held-out fold (for plots)
train_idx, test_idx = list(skf.split(X, y))[-1]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)

# %% Plot 1: CV scores comparison
fig, ax = plt.subplots(figsize=(8, 4))
folds = np.arange(1, N_FOLDS + 1)
width = 0.35

ax.bar(
    folds - width / 2,
    xgb_scores,
    width,
    label=f"XGBoost (mean={xgb_scores.mean():.4f})",
    color="#2D6A9F",
)
ax.bar(
    folds + width / 2,
    rf_scores,
    width,
    label=f"Random Forest (mean={rf_scores.mean():.4f})",
    color="#E07B3B",
)
ax.axhline(xgb_scores.mean(), color="#2D6A9F", linestyle="--", linewidth=1.2, alpha=0.7)
ax.axhline(rf_scores.mean(), color="#E07B3B", linestyle="--", linewidth=1.2, alpha=0.7)

ax.set_xlabel("Fold")
ax.set_ylabel("ROC-AUC")
ax.set_title("XGBoost vs Random Forest — Stratified 5-Fold CV ROC-AUC")
ax.set_xticks(folds)
ax.legend()
ax.set_ylim(0.7, 1.0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS / "cv_scores.png", dpi=150)
plt.close()
print("Saved: cv_scores.png")

# %% Plot 2: ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2D6A9F", lw=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.1, color="#2D6A9F")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost on Titanic (held-out fold)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS / "roc_curve.png", dpi=150)
plt.close()
print("Saved: roc_curve.png")

# %% Plot 3: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Did Not Survive", "Survived"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — XGBoost (default threshold = 0.50)")

plt.tight_layout()
plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# %% Plot 4: Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(
    recall,
    precision,
    color="#2D6A9F",
    lw=2,
    label=f"XGBoost (AP = {avg_precision:.4f})",
)
ax.axhline(
    y.mean(),
    color="gray",
    linestyle="--",
    lw=1,
    label=f"Random baseline (AP = {y.mean():.4f})",
)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — XGBoost on Titanic")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS / "precision_recall_curve.png", dpi=150)
plt.close()
print("Saved: precision_recall_curve.png")

# %% Summary
print("\nClassification Report (threshold = 0.50)")
print(
    classification_report(y_test, y_pred, target_names=["Did Not Survive", "Survived"])
)
print(f"ROC-AUC (held-out fold): {roc_auc:.4f}")
print(f"CV ROC-AUC (5-fold mean): {xgb_scores.mean():.4f}")
