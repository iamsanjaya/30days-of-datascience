"""
06_final_eval_shap.py — Capstone 1: Home Credit Default Risk

Final evaluation of the tuned LightGBM pipeline saved by
05_tune_best_model_optuna.py. Does NOT retrain or re-run Optuna — loads
the saved joblib pipeline directly, so this is a fast script by comparison.

Reproduces the IDENTICAL train/test split used in 04 and 05 (same source
parquet, same train_test_split params) so the held-out test set used here
is genuinely the same one the model never saw during tuning.

Deliverables:
  1. Bootstrap confidence interval on held-out test ROC-AUC.
  2. SHAP TreeExplainer explanations for the top 5 features (summary +
     bar plot), using the real post-ColumnTransformer feature names.
"""

# %% Imports
from typing import cast

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import config
from utils.plot_utils import save_and_close

SHAP_OUTPUT_DIR = config.OUTPUT_DIR / "shap"
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
TOP_N_FEATURES = 5

# %% Load engineered features — SAME source as 04 and 05
df = pd.read_parquet(config.OUTPUT_DIR / "application_train_features.parquet")
print(f"[06_eval] Loaded {df.shape}")

X = df.drop(columns=[config.ID_COL, config.TARGET_COL])
y = df[config.TARGET_COL]

# %% Reproduce the IDENTICAL split from 04/05 (same params, same data -> same split)
# Explicit cast on the unpacked outputs: train_test_split's type stubs are
# generic enough that pyright/Pylance can infer plain `list` here instead
# of DataFrame/Series, which then breaks .shape access and any typed
# function call downstream (e.g. bootstrap_auc_ci's y_true: pd.Series
# parameter) — same class of stub-looseness issue as the shap_values.values
# cast above. The objects are genuinely DataFrame/Series at runtime.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)
X_train = cast(pd.DataFrame, X_train)
X_test = cast(pd.DataFrame, X_test)
y_train = cast(pd.Series, y_train)
y_test = cast(pd.Series, y_test)
print(f"[06_eval] Train: {X_train.shape}, Test (held-out): {X_test.shape}")

# %% Load the tuned pipeline saved by 05 — no retraining
model_path = config.MODEL_DIR / "lightgbm_tuned.joblib"
if not model_path.exists():
    raise FileNotFoundError(
        f"{model_path} not found. Run 05_tune_best_model_optuna.py first."
    )

pipeline = joblib.load(model_path)
preprocessor = pipeline.named_steps["preprocessor"]
lgbm_model = pipeline.named_steps["model"]
print(
    f"[06_eval] Loaded pipeline: {type(lgbm_model).__name__} "
    f"({lgbm_model.n_features_in_} features after preprocessing)"
)

# %% Point-estimate test ROC-AUC (sanity check vs. 05's reported 0.7712)
test_proba = pipeline.predict_proba(X_test)[:, 1]
point_auc = float(roc_auc_score(y_test, test_proba))
print(f"\n[06_eval] Point-estimate test ROC-AUC: {round(point_auc, 4)}")


# %% Bootstrap confidence interval on test ROC-AUC
def bootstrap_auc_ci(
    proba: np.ndarray,
    y_true: pd.Series,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI_LEVEL,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, float, float, np.ndarray]:
    """
    Resample (proba, y_true) pairs together with replacement and recompute
    ROC-AUC each time. Resampling the already-predicted probabilities
    (rather than re-predicting inside the loop) keeps this fast — no need
    to re-run the pipeline 1000 times since predictions don't change.
    """
    rng = np.random.RandomState(random_state)
    y_arr = np.array(y_true, dtype=int)
    n = len(y_arr)
    scores = []

    for _ in range(n_bootstrap):
        idx = resample(
            np.arange(n), replace=True, n_samples=n, random_state=rng.randint(1_000_000)
        )
        y_sample = y_arr[idx]
        if len(np.unique(y_sample)) < 2:
            continue  # degenerate resample (single class) -> ROC-AUC undefined
        scores.append(roc_auc_score(y_sample, proba[idx]))

    scores = np.array(scores)
    lower = float(np.percentile(scores, (1 - ci) / 2 * 100))
    upper = float(np.percentile(scores, (1 - (1 - ci) / 2) * 100))
    mean_score = float(np.mean(scores))
    return mean_score, lower, upper, scores


print(
    f"\n[06_eval] Running {N_BOOTSTRAP} bootstrap resamples for {int(CI_LEVEL * 100)}% CI..."
)
boot_mean, ci_lower, ci_upper, boot_scores = bootstrap_auc_ci(test_proba, y_test)
print(
    f"[06_eval] Bootstrap ROC-AUC: {round(boot_mean, 4)} "
    f"(95% CI: [{round(ci_lower, 4)}, {round(ci_upper, 4)}])"
)

ci_results = pd.DataFrame(
    {
        "metric": [
            "test_roc_auc_point_estimate",
            "bootstrap_mean",
            f"ci_lower_{int(CI_LEVEL * 100)}",
            f"ci_upper_{int(CI_LEVEL * 100)}",
            "n_bootstrap",
        ],
        "value": [
            round(point_auc, 4),
            round(boot_mean, 4),
            round(ci_lower, 4),
            round(ci_upper, 4),
            len(boot_scores),
        ],
    }
)
ci_path = config.OUTPUT_DIR / "bootstrap_ci_results.csv"
ci_results.to_csv(ci_path, index=False)
print(f"[06_eval] Saved CI results -> {ci_path}")

# %% SHAP — TreeExplainer on the raw LGBMClassifier, fed transformed features
print("\n[06_eval] Transforming test features through the fitted preprocessor...")
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# ColumnTransformer output can be sparse (OneHotEncoder default) — SHAP's
# TreeExplainer wants a dense array for the summary/bar plots below.
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

print("[06_eval] Running SHAP TreeExplainer (exact, fast for tree models)...")
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer(X_test_transformed_df)

# Binary classification: some SHAP/LightGBM version combos return a third
# axis for [class_0, class_1] contributions — normalize to the positive class.
# Explicit np.asarray() cast: shap's type stubs are loose enough that
# Pylance can infer shap_values.values as a plain list rather than
# ndarray, which then breaks .ndim and the [:, :, 1] multi-axis slice
# below — this is a static-typing-only issue, not a runtime bug (the
# underlying object is genuinely an ndarray at runtime).
shap_array: np.ndarray = np.asarray(shap_values.values)
if shap_array.ndim == 3:
    shap_array = shap_array[:, :, 1]

mean_abs_shap = np.abs(shap_array).mean(axis=0)
importance_df = (
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
top_features = importance_df.head(TOP_N_FEATURES)

print(f"\n[06_eval] Top {TOP_N_FEATURES} features by mean |SHAP value|:")
print(top_features.to_string(index=False))

top_features.to_csv(config.OUTPUT_DIR / "shap_top5_features.csv", index=False)

# %% SHAP summary (beeswarm) plot — top 5 features
# shap.summary_plot() manages its own figure internally (calls plt.gcf()
# under the hood) rather than drawing onto a figure handle passed in by
# reference — so the figure must be captured via plt.gcf() AFTER the plot
# call, not created beforehand and assumed to be reused.
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_array, X_test_transformed_df, show=False, max_display=TOP_N_FEATURES
)
fig_summary = plt.gcf()
save_and_close(fig_summary, SHAP_OUTPUT_DIR, "01_shap_summary_top5.png")

# %% SHAP bar plot — mean |SHAP value| per feature
fig_bar, ax = plt.subplots(figsize=(8, 5))
ax.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
ax.set_xlabel("Mean |SHAP value| (impact on model output)")
ax.set_title(f"Top {TOP_N_FEATURES} Features by SHAP Importance")
fig_bar.tight_layout()
save_and_close(fig_bar, SHAP_OUTPUT_DIR, "02_shap_bar_top5.png")

print(f"\n[06_eval] Done. CI results and SHAP outputs saved under {config.OUTPUT_DIR}")
