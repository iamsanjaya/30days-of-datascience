"""
05_tune_best_model_optuna.py — Capstone 1: Home Credit Default Risk

Tunes LightGBM (the winner from 04_train_baseline_models.py: CV ROC-AUC
0.7617 vs 0.7489 for Logistic Regression and 0.7223 for Random Forest)
using Optuna's TPE sampler, logging every trial to MLflow (SQLite backend,
matching the established mlflow.db setup).

Uses the SAME train/test split as 04 (identical random_state + test_size
on the same data reproduces it deterministically) so the held-out test set
stays genuinely unseen throughout tuning, not just during the final model
comparison.

Search uses OPTUNA_CV_FOLDS (3) rather than the full N_CV_FOLDS (5) — 50
trials x 5 folds x LightGBM is an expensive way to buy marginal precision
during search, when all the search loop needs is a stable enough signal to
rank hyperparameter combinations against each other. The final best params
get re-evaluated with the full N_CV_FOLDS for the number that actually gets
reported.
"""

# %% Imports
from typing import cast

import lightgbm as lgb
import mlflow
import mlflow.sklearn as mlflow_sklearn
import optuna
import optuna.visualization.matplotlib as optuna_viz
import pandas as pd
import joblib

from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
from utils.plot_utils import save_and_close

TUNING_OUTPUT_DIR = config.OUTPUT_DIR / "optuna"

# %% Load engineered features — same source as 04
df = pd.read_parquet(config.OUTPUT_DIR / "application_train_features.parquet")
print(f"[05_tune] Loaded {df.shape}")

X = df.drop(columns=[config.ID_COL, config.TARGET_COL])
y = df[config.TARGET_COL]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

# %% Reproduce the IDENTICAL split from 04 (same params, same data -> same split)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y,
)
print(f"[05_tune] Train: {X_train.shape}, Test: {X_test.shape}")

# %% Shared preprocessing (identical to 04's LightGBM branch — no scaler needed)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# %% MLflow setup — SQLite backend, matching the established mlflow.db pattern
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
print(f"[05_tune] MLflow tracking URI: {config.MLFLOW_TRACKING_URI}")

# %% Optuna objective
search_cv = StratifiedKFold(
    n_splits=config.OPTUNA_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
)


def objective(trial: optuna.Trial) -> float:
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                lgb.LGBMClassifier(
                    **params,
                    bagging_freq=1,  # required for `subsample` to actually take effect
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=search_cv,
        scoring=config.PRIMARY_METRIC,
        n_jobs=1,
    )
    cv_mean = float(scores.mean())
    cv_std = float(scores.std())

    with mlflow.start_run(nested=False):
        mlflow.log_params(params)
        mlflow.log_metric("cv_mean_auc", cv_mean)
        mlflow.log_metric("cv_std_auc", cv_std)

    return cv_mean


# %% Run the study (persistent storage — resumable if the session dies)
study = optuna.create_study(
    study_name=config.OPTUNA_STUDY_NAME,
    storage=config.OPTUNA_STORAGE_URI,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
)

n_completed = len(study.trials)
n_remaining = max(config.OPTUNA_N_TRIALS - n_completed, 0)

if n_completed > 0:
    print(
        f"[05_tune] Resuming existing study: {n_completed} trial(s) already "
        f"completed, running {n_remaining} more to reach target of "
        f"{config.OPTUNA_N_TRIALS}."
    )

if n_remaining > 0:
    study.optimize(objective, n_trials=n_remaining)
else:
    print(
        f"[05_tune] Study already has {n_completed} trials, meeting or "
        f"exceeding the target of {config.OPTUNA_N_TRIALS} — skipping further optimization."
    )

print(
    f"\n[05_tune] Best trial value (CV ROC-AUC, {config.OPTUNA_CV_FOLDS}-fold): {study.best_value:.4f}"
)
print(f"[05_tune] Best params:\n{study.best_params}")

# %% Optuna diagnostic plots
fig = cast(Figure, optuna_viz.plot_optimization_history(study).figure)
save_and_close(fig, TUNING_OUTPUT_DIR, "01_optimization_history.png")

fig = cast(Figure, optuna_viz.plot_param_importances(study).figure)
save_and_close(fig, TUNING_OUTPUT_DIR, "02_param_importances.png")

fig = cast(Figure, optuna_viz.plot_parallel_coordinate(study).figure)
save_and_close(fig, TUNING_OUTPUT_DIR, "03_parallel_coordinate.png")

# %% Re-evaluate best params with the FULL N_CV_FOLDS for the reported number
best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            lgb.LGBMClassifier(
                **study.best_params,
                bagging_freq=1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]
)

full_cv = StratifiedKFold(
    n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
)
final_cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train,
    cv=full_cv,
    scoring=config.PRIMARY_METRIC,
    n_jobs=-1,
)
final_cv_mean = float(final_cv_scores.mean())
final_cv_std = float(final_cv_scores.std())
print(
    f"\n[05_tune] Tuned model — full {config.N_CV_FOLDS}-fold CV ROC-AUC: "
    f"{final_cv_mean:.4f} +/- {final_cv_std:.4f}"
)
print(
    f"[05_tune] Baseline (untuned) LightGBM CV ROC-AUC was 0.7617 — "
    f"tuning {'improved' if final_cv_mean > 0.7617 else 'did not improve'} on it."
)

# %% Fit on full training set, evaluate on held-out test set
best_pipeline.fit(X_train, y_train)
test_proba = best_pipeline.predict_proba(X_test)[:, 1]
test_auc = float(roc_auc_score(y_test, test_proba))
print(f"[05_tune] Tuned model — held-out test ROC-AUC: {test_auc:.4f}")

# %% Save the tuned pipeline for 06_final_eval_shap.py — BEFORE any MLflow
# logging, so this critical artifact exists regardless of whether MLflow's
# model-logging step succeeds. 06 depends on this file; it does not
# depend on MLflow at all.

model_path = config.MODEL_DIR / "lightgbm_tuned.joblib"
joblib.dump(best_pipeline, model_path)
print(f"[05_tune] Saved tuned pipeline to {model_path}")

# %% Log the final tuned run to MLflow as the headline result
# log_model is wrapped separately and non-fatally: MLflow's skops-based
# serializer can refuse to log certain model types (e.g. LightGBM's
# internal Booster class) as an "untrusted type" security measure. That's
# a tracking/logging nicety, not something that should block the pipeline
# — the actual model is already safely saved above via joblib.
with mlflow.start_run(run_name="best_tuned_lightgbm"):
    mlflow.log_params(study.best_params)
    mlflow.log_metric("cv_mean_auc_full_folds", final_cv_mean)
    mlflow.log_metric("cv_std_auc_full_folds", final_cv_std)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.set_tag("model_type", "lightgbm_tuned")
    try:
        mlflow_sklearn.log_model(
            best_pipeline,
            "model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )
    except Exception as e:
        print(
            f"[05_tune] WARNING: MLflow model logging failed ({e}). "
            "Params/metrics were still logged. The joblib file (saved above) "
            "is the one 06_final_eval_shap.py actually depends on, so this "
            "failure does not block the pipeline."
        )
