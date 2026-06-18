# 03_grid_search.py — Day 15
# GridSearchCV over the FULL pipeline — param names use the
# "step__param" double-underscore syntax so the grid search
# tunes XGBoost params through the pipeline, not around it.
# This is the correct way to tune: grid search never sees test data.

# %%
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
import xgboost as xgb

from config import XGB_PARAMS
from pipeline_factory import build_pipeline as _build
from config import (
    DATA_PATH,
    TARGET,
    DROP_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    SCORING,
    GRID_PARAM_GRID,
    OUTPUT_DIR,
)
from pipeline_factory import build_pipeline

# ── 1. Load & split ───────────────────────────────────────────────────────────
# %%
df = pd.read_csv(DATA_PATH).drop(columns=DROP_COLS)
all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[all_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ── 2. Build base pipeline ────────────────────────────────────────────────────
# %%
# XGBRegressor with minimal defaults — GridSearch will override the params
# listed in GRID_PARAM_GRID via the "model__" prefix.
base_xgb = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
pipeline = build_pipeline(base_xgb)

# ── 3. GridSearchCV over the full pipeline ────────────────────────────────────
# %%
# CRITICAL: param keys use "model__" prefix because the estimator is named
# "model" inside the pipeline. The grid search refits the ENTIRE pipeline
# (imputation + scaling + OHE + XGB) per fold — no preprocessing leakage.

print(f"Grid: {GRID_PARAM_GRID}")
print(
    f"Combinations: {len(GRID_PARAM_GRID['model__n_estimators']) * len(GRID_PARAM_GRID['model__max_depth']) * len(GRID_PARAM_GRID['model__learning_rate'])}"
)
print(f"Total fits: combinations × {CV_FOLDS} folds\n")

grid_search = GridSearchCV(
    pipeline,
    param_grid=GRID_PARAM_GRID,
    cv=CV_FOLDS,
    scoring=SCORING,
    refit=True,  # refit best params on full X_train after search
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

# ── 4. Results ────────────────────────────────────────────────────────────────
# %%
print("\n── GridSearchCV Results ──")
print(f"Best params : {grid_search.best_params_}")
print(f"Best CV RMSE: ${-grid_search.best_score_:,.0f}")

# All combinations ranked
cv_df = pd.DataFrame(grid_search.cv_results_)
cv_df["mean_rmse"] = -cv_df["mean_test_score"]
cv_df["std_rmse"] = cv_df["std_test_score"]

display_cols = [c for c in cv_df.columns if c.startswith("param_")] + [
    "mean_rmse",
    "std_rmse",
    "rank_test_score",
]
ranked = cv_df[display_cols].sort_values("rank_test_score")
print("\n── All combinations (ranked) ──")
print(ranked.to_string(index=False))

# ── 5. Test set evaluation with best estimator ────────────────────────────────
# %%
y_pred = grid_search.best_estimator_.predict(X_test)
test_rmse = root_mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n── Best Estimator on Held-Out Test ──")
print(f"Test RMSE : ${test_rmse:>10,.0f}")
print(f"Test R²   : {test_r2:.4f}")

# ── 6. Default vs tuned comparison ───────────────────────────────────────────
# %%
# Refit default pipeline for fair comparison


default_pipeline = _build(xgb.XGBRegressor(**XGB_PARAMS))
default_pipeline.fit(X_train, y_train)
default_pred = default_pipeline.predict(X_test)
default_rmse = root_mean_squared_error(y_test, default_pred)

improvement = default_rmse - test_rmse
print("\n── Default vs GridSearch ──")
print(f"Default RMSE : ${default_rmse:>10,.0f}")
print(f"Tuned   RMSE : ${test_rmse:>10,.0f}")
print(f"Improvement  : ${improvement:>10,.0f}  ({improvement/default_rmse*100:.2f}%)")

# ── 7. Plot: GridSearch RMSE heatmap (n_estimators × max_depth per lr) ────────
# %%
for lr in GRID_PARAM_GRID["model__learning_rate"]:
    subset = cv_df[cv_df["param_model__learning_rate"] == lr].copy()
    pivot = subset.pivot(
        index="param_model__max_depth",
        columns="param_model__n_estimators",
        values="mean_rmse",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title(f"GridSearch CV RMSE Heatmap  |  lr={lr}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"${pivot.values[i, j]:,.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    plt.colorbar(im, ax=ax, label="CV RMSE ($)")
    plt.tight_layout()
    fname = f"03_gridsearch_heatmap_lr{str(lr).replace('.','')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved: outputs/{fname}")

# ── 8. Plot: Default vs Tuned RMSE comparison ─────────────────────────────────
# %%
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Default\n(config.py)", "GridSearch Best"]
values = [default_rmse / 1000, test_rmse / 1000]
colors = ["#5b8db8", "#6aaa64" if test_rmse < default_rmse else "#d9534f"]
bars = ax.bar(labels, values, color=colors, width=0.4)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"${val*1000:,.0f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_ylabel("Test RMSE ($000s)")
ax.set_title(
    "Default vs GridSearch Best — Test Set RMSE\n"
    "(Single 80/20 split · high variance at n=586)"
)
ax.set_ylim(0, max(values) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_default_vs_tuned.png"), dpi=150)
plt.close()
print("Saved: outputs/03_default_vs_tuned.png")

print("\n03_grid_search.py complete.")
