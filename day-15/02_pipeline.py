# 02_pipeline.py — Day 15
# Build the full sklearn Pipeline (preprocessor + XGBoost).
# Evaluate with 5-fold CV, then score the held-out test set.
# Prints RMSE and R² — no leakage because fit() only sees X_train.

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb

from config import (
    DATA_PATH,
    TARGET,
    DROP_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    XGB_PARAMS,
    OUTPUT_DIR,
)
from pipeline_factory import build_pipeline

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & split ───────────────────────────────────────────────────────────
# %%
df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[all_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── 2. Build pipeline ─────────────────────────────────────────────────────────
# %%
# The pipeline encapsulates: imputation → scaling → OHE → XGBoost
# fit() on X_train means the scaler and imputer never see test data.
# That is the central guarantee of a Pipeline.

model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
pipeline = build_pipeline(model_xgb)

print("Pipeline steps:")
for name, step in pipeline.steps:
    print(f"  {name}: {type(step).__name__}")

# ── 3. Cross-validation on train set ──────────────────────────────────────────
# %%
# cross_validate internally refits the pipeline on each fold's train split.
# The scaler/imputer are fit fresh per fold — no leakage across folds.
print(f"\nRunning {CV_FOLDS}-fold CV on training set...")

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=CV_FOLDS,
    scoring=["neg_root_mean_squared_error", "r2"],
    return_train_score=True,
    n_jobs=-1,
)

cv_rmse_val = -cv_results["test_neg_root_mean_squared_error"]
cv_rmse_train = -cv_results["train_neg_root_mean_squared_error"]
cv_r2_val = cv_results["test_r2"]

print(f"\n── {CV_FOLDS}-Fold CV Results (train set) ──")
print(f"Val   RMSE  : ${cv_rmse_val.mean():>10,.0f}  ± ${cv_rmse_val.std():,.0f}")
print(f"Train RMSE  : ${cv_rmse_train.mean():>10,.0f}  ± ${cv_rmse_train.std():,.0f}")
print(f"Val   R²    : {cv_r2_val.mean():.4f}  ± {cv_r2_val.std():.4f}")

gap = cv_rmse_train.mean() - cv_rmse_val.mean()
print(
    f"Train-Val RMSE gap: ${gap:,.0f}  {'(overfit signal)' if abs(gap) > 5000 else '(acceptable)'}"
)

# ── 4. Final fit on full train, evaluate on held-out test ─────────────────────
# %%
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

test_rmse = root_mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n── Held-Out Test Set ──")
print(f"Test RMSE : ${test_rmse:>10,.0f}")
print(f"Test R²   : {test_r2:.4f}")
print(
    f"Mean SalePrice: ${y_test.mean():,.0f}  →  RMSE is {test_rmse/y_test.mean()*100:.1f}% of mean price"
)

# ── 5. Top feature importances (XGBoost gain) ────────────────────────────────
# %%
# Recover feature names after OHE expansion
ohe_features: list[str] = list(
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["ohe"]
    .get_feature_names_out(CATEGORICAL_FEATURES)
)
all_transformed_features = NUMERIC_FEATURES + ohe_features

importances = pd.Series(
    pipeline.named_steps["model"].feature_importances_,
    index=all_transformed_features,
).sort_values(ascending=False)

print("\n── Top 15 Features by XGBoost Gain ──")
print(importances.head(15).to_string())

# ── 6. Plot: Top 15 feature importances ──────────────────────────────────────
# %%
top15 = importances.head(15).sort_values()

x = np.asarray(top15.index)
y = np.asarray(top15.values)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(x, y, color="#5b8db8")
ax.set_xlabel("XGBoost Gain (importance)")
ax.set_title("Top 15 Features by XGBoost Gain\nAmes Housing Pipeline")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_feature_importance.png"), dpi=150)
plt.close()
print("\nSaved: outputs/02_feature_importance.png")

# ── 7. Plot: CV fold RMSE (train vs val per fold) ─────────────────────────────
# %%
folds = range(1, CV_FOLDS + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(folds, cv_rmse_train / 1000, marker="o", label="Train RMSE", color="#6aaa64")
ax.plot(folds, cv_rmse_val / 1000, marker="o", label="Val RMSE", color="#d9534f")
ax.axhline(
    cv_rmse_train.mean() / 1000,
    linestyle="--",
    color="#6aaa64",
    alpha=0.5,
    linewidth=0.8,
)
ax.axhline(
    cv_rmse_val.mean() / 1000, linestyle="--", color="#d9534f", alpha=0.5, linewidth=0.8
)
ax.set_xlabel("Fold")
ax.set_ylabel("RMSE ($000s)")
ax.set_title(
    "5-Fold CV: Train vs Validation RMSE per Fold\n"
    f"Mean Val RMSE: ${cv_rmse_val.mean():,.0f}  |  Gap: ${(cv_rmse_val - cv_rmse_train).mean():,.0f}"
)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_cv_fold_rmse.png"), dpi=150)
plt.close()
print("Saved: outputs/02_cv_fold_rmse.png")

print("\n02_pipeline.py complete.")
