# optuna_tuner.py — Day 16
# Runs the main Optuna study: 55 Bayesian trials, each logged to MLflow.
# Returns the completed study object so plots.py can consume it downstream.

import warnings

import mlflow
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from config import (
    CV_FOLDS,
    CV_SCORING,
    MLFLOW_EXPERIMENT_NAME,
    OPTUNA_DIRECTION,
    OPTUNA_N_TRIALS,
    OPTUNA_STUDY_NAME,
    RANDOM_STATE,
    XGB_FIXED_PARAMS,
    XGB_SEARCH_SPACE,
)

# Suppress Optuna's per-trial INFO logs — we emit our own summary lines.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def _build_params(trial: optuna.Trial) -> dict:
    """Translate the config search space into a concrete parameter dict for one trial."""
    sp = XGB_SEARCH_SPACE
    return {
        "n_estimators": trial.suggest_int("n_estimators", *sp["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *sp["learning_rate"], log=True
        ),
        "subsample": trial.suggest_float("subsample", *sp["subsample"]),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *sp["colsample_bytree"]
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *sp["min_child_weight"]
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", *sp["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *sp["reg_lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *sp["gamma"]),
    }


def _make_objective(X_train, y_train):
    """Return an Optuna objective closure that captures the training data."""

    def objective(trial: optuna.Trial) -> float:
        params = {**_build_params(trial), **XGB_FIXED_PARAMS}
        model = XGBRegressor(**params)

        # Cross-val returns negative RMSE (sklearn convention); flip sign for true RMSE.
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring=CV_SCORING,
            n_jobs=-1,
        )
        rmse = -scores.mean()

        # Log each trial as a child run inside the parent MLflow experiment.
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"})
            mlflow.log_metric("cv_rmse", rmse)
            mlflow.log_metric("cv_rmse_std", scores.std())

        if trial.number % 10 == 0 or trial.number == OPTUNA_N_TRIALS - 1:
            print(f"  Trial {trial.number:3d} | RMSE: {rmse:,.0f}")

        return rmse

    return objective


def run_full_study(X_train, y_train) -> optuna.Study:
    """Run the main 55-trial Optuna study and return the completed study object.

    All trials are logged under MLFLOW_EXPERIMENT_NAME.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction=OPTUNA_DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    print(f"\n[optuna_tuner] Starting full study — {OPTUNA_N_TRIALS} trials")

    with mlflow.start_run(run_name="full-study-parent"):
        study.optimize(
            _make_objective(X_train, y_train),
            n_trials=OPTUNA_N_TRIALS,
            show_progress_bar=False,
        )

        # Log aggregate best-trial metrics on the parent run.
        best = study.best_trial

    assert (
        best.value is not None
    ), "Optuna returned None best value (invalid study state)."

    best_value: float = best.value

    mlflow.log_metric("best_cv_rmse", best_value)
    mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

    print(f"\n[optuna_tuner] Best RMSE : {best.value:,.0f}")
    print(f"[optuna_tuner] Best params: {best.params}")
    return study


def get_best_model(study: optuna.Study) -> XGBRegressor:
    """Instantiate an XGBRegressor with the best parameters from the study."""
    best_params = {**study.best_trial.params, **XGB_FIXED_PARAMS}
    return XGBRegressor(**best_params)
