import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from typing import Any, cast

from config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
    TARGET_COL,
    DROP_COLS,
    XGB_PARAMS,
    CV_FOLDS,
    RANDOM_STATE,
    ABLATION_RESULTS_PATH,
    ABLATION_RETENTION_THRESHOLD,
)
from utils.evaluation import cv_rmsle
from utils.encoding import encode_categoricals

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(PROCESSED_DATA_PATH)

y = df[TARGET_COL]

drop_cols = [TARGET_COL] + [c for c in DROP_COLS if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")

X = X.fillna("Unknown")
X = encode_categoricals(X)

# -----------------------------
# Baseline CV
# -----------------------------
model = XGBRegressor(**XGB_PARAMS)

print("FULL MODEL")
full = cv_rmsle(model, X, y, CV_FOLDS, RANDOM_STATE)
print(full)

full_rmsle = full["mean"]
threshold = full_rmsle / ABLATION_RETENTION_THRESHOLD

# -----------------------------
# Permutation Importance
# -----------------------------
log_y = np.log1p(y)

perm_model = XGBRegressor(**XGB_PARAMS)
perm_model.fit(X, log_y)

perm = permutation_importance(
    perm_model,
    X,
    log_y,
    scoring="neg_root_mean_squared_error",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

perm_result = cast(Any, perm)

importance = pd.Series(
    np.asarray(perm_result.importances_mean),
    index=X.columns,
    dtype=float,
).sort_values(ascending=False)

# -----------------------------
# Greedy Backward Ablation
# -----------------------------
features = list(importance.index)
current_features = features.copy()

ablation_log = []

print("\nABLATION START")

step = 0

for f in reversed(features):  # least important first
    if len(current_features) <= 5:
        break

    candidate = [c for c in current_features if c != f]

    model = XGBRegressor(**XGB_PARAMS)
    res = cv_rmsle(model, X[candidate], y, CV_FOLDS, RANDOM_STATE)

    step += 1

    row = {
        "step": step,
        "removed": f,
        "features_remaining": len(candidate),
        "rmsle_mean": res["mean"],
        "rmsle_std": res["std"],
    }

    ablation_log.append(row)

    print(f"{step:03d} | removed={f} | rmsle={res['mean']:.5f}")

    if res["mean"] > threshold:
        break

    current_features = candidate

ablation_df = pd.DataFrame(ablation_log)

# -----------------------------
# Save artifacts
# -----------------------------
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ablation_df.to_csv(ABLATION_RESULTS_PATH, index=False)

# -----------------------------
# Ablation Curve
# -----------------------------
plt.figure(figsize=(10, 5))

if not ablation_df.empty:
    plt.plot(
        ablation_df["features_remaining"],
        ablation_df["rmsle_mean"],
        marker="o",
    )

plt.xlabel("Features Remaining")
plt.ylabel("CV RMSLE")
plt.title("Anti-Feature Ablation Curve")
plt.gca().invert_xaxis()
plt.grid(alpha=0.3)

plot_path = PLOTS_DIR / "ablation_curve.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"[SAVED] ablation_log -> {ABLATION_RESULTS_PATH}")
print(f"[SAVED] ablation_curve -> {plot_path}")

# -----------------------------
# Top Features
# -----------------------------
print("\nTOP FEATURES")
print(importance.head(20))
