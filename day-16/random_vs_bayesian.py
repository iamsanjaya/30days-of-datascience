# random_vs_bayesian.py — Day 16 Out-of-Box Challenge
#
# Production-safe comparison: Random vs Bayesian (TPE)
# Fixes included:
# - MLflow run scope correctness
# - stable best-so-far tracking
# - float safety for metrics
# - deterministic trial logging
# - no cross-trial state leakage

import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from config import (
    BAYESIAN_SEARCH_RUN_NAME,
    COMPARISON_N_TRIALS,
    CV_FOLDS,
    CV_SCORING,
    MLFLOW_EXPERIMENT_NAME,
    RANDOM_SEARCH_RUN_NAME,
    RANDOM_STATE,
    XGB_FIXED_PARAMS,
    XGB_SEARCH_SPACE,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def _suggest_params(trial: optuna.Trial) -> dict:
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


def _run_comparison_study(
    X_train,
    y_train,
    sampler: optuna.samplers.BaseSampler,
    run_name: str,
) -> pd.DataFrame:

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    records: list[dict] = []
    best_so_far_values: list[float] = []

    def objective(trial: optuna.Trial) -> float:
        params = {**_suggest_params(trial), **XGB_FIXED_PARAMS}

        model = XGBRegressor(**params)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring=CV_SCORING,
            n_jobs=-1,
        )

        rmse = float(-scores.mean())

        current_best = min(best_so_far_values) if best_so_far_values else float("inf")

        if rmse < current_best:
            mode = "exploitation"
            best_so_far_values.append(rmse)
        else:
            mode = "exploration"
            best_so_far_values.append(current_best)

        records.append(
            {
                "trial": int(trial.number),
                "rmse": rmse,
                "best_so_far": best_so_far_values[-1],
                "mode": mode,
            }
        )

        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    with mlflow.start_run(run_name=run_name, nested=True):
        study.optimize(
            objective,
            n_trials=COMPARISON_N_TRIALS,
            show_progress_bar=False,
        )

        mlflow.log_metric("best_rmse", float(study.best_value))
        mlflow.log_metric("n_trials", int(COMPARISON_N_TRIALS))

    df = pd.DataFrame(records)

    print(
        f"  [{run_name}] Best RMSE after {COMPARISON_N_TRIALS} trials: "
        f"{float(study.best_value):,.0f}"
    )

    return df


def run_comparison(X_train, y_train) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[random_vs_bayesian] Running Random Search...")

    random_df = _run_comparison_study(
        X_train,
        y_train,
        sampler=optuna.samplers.RandomSampler(seed=RANDOM_STATE),
        run_name=RANDOM_SEARCH_RUN_NAME,
    )

    print("[random_vs_bayesian] Running Bayesian Search (TPE)...")

    bayesian_df = _run_comparison_study(
        X_train,
        y_train,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        run_name=BAYESIAN_SEARCH_RUN_NAME,
    )

    random_best = float(random_df["rmse"].min())
    bayesian_best = float(bayesian_df["rmse"].min())

    winner = "Bayesian (TPE)" if bayesian_best < random_best else "Random"

    print("\n  ── Comparison Summary ─────────────────────────────")
    print(f"  Random   best RMSE : {random_best:,.0f}")
    print(f"  Bayesian best RMSE : {bayesian_best:,.0f}")
    print(f"  Winner            : {winner}")

    return random_df, bayesian_df
