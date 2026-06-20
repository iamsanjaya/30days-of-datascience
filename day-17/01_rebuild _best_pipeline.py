"""
01_rebuild_best_pipeline.py — Day 17

Rebuilds Day 15's pipeline exactly as it stands (raw Ames features only —
Day 14's engineered features were never folded in; see MODEL_CARD.md).

Day 15's best GridSearchCV hyperparameters were never persisted, so this
script re-runs the same 12-combo grid to rediscover them, then freezes and
persists the winning pipeline for scripts 02-05 to reuse without refitting.

Handles the cross-day config.py collision: Day 15's pipeline_factory.py
does `from config import (...)` expecting ITS OWN config.py, not Day 17's.
We temporarily swap sys.modules["config"] while importing pipeline_factory,
then restore Day 17's own config.

y is kept as a pandas Series (not a bare ndarray) throughout, since
02_distribution_shift_test.py indexes it with a boolean mask and calls
`.values` on slices.
"""

import importlib.util
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

import config


def _load_module_with_local_config(module_name: str, module_path, config_path):
    """
    Import `module_path` as `module_name` while making `from config import X`
    inside it resolve to `config_path`, not whatever "config" module is
    already cached in sys.modules (Day 17's own config.py in our case).

    Restores the original sys.modules["config"] afterwards so the rest of
    this script keeps using Day 17's config as normal.
    """
    original_config = sys.modules.get("config")

    cfg_spec = importlib.util.spec_from_file_location("config", config_path)
    if cfg_spec is None or cfg_spec.loader is None:
        raise ImportError(f"Invalid config module spec: {config_path}")

    cfg_module = importlib.util.module_from_spec(cfg_spec)
    sys.modules["config"] = cfg_module
    cfg_spec.loader.exec_module(cfg_module)

    try:
        target_spec = importlib.util.spec_from_file_location(module_name, module_path)
        if target_spec is None or target_spec.loader is None:
            raise ImportError(f"Invalid target module spec: {module_path}")

        target_module = importlib.util.module_from_spec(target_spec)
        target_spec.loader.exec_module(target_module)
    finally:
        if original_config is not None:
            sys.modules["config"] = original_config
        else:
            sys.modules.pop("config", None)

    return target_module


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(config.AMES_CSV_PATH)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return df


def bootstrap_rmse_ci(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Bootstrap the test-set residuals to get an RMSE confidence interval."""
    y_true_arr = y_true.to_numpy()
    rng = np.random.RandomState(config.RANDOM_STATE)
    n = len(y_true_arr)
    boot_rmses = []
    for _ in range(config.BOOTSTRAP_ITERATIONS):
        idx = rng.randint(0, n, size=n)
        boot_rmses.append(np.sqrt(mean_squared_error(y_true_arr[idx], y_pred[idx])))
    boot_rmses = np.array(boot_rmses)
    return {
        "point_estimate": float(np.sqrt(mean_squared_error(y_true_arr, y_pred))),
        "ci_lower": float(np.percentile(boot_rmses, config.CI_LOWER_PCT)),
        "ci_upper": float(np.percentile(boot_rmses, config.CI_UPPER_PCT)),
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true_arr = y_true.to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
    rmsle = float(
        np.sqrt(
            mean_squared_error(np.log1p(y_true_arr), np.log1p(np.clip(y_pred, 0, None)))
        )
    )
    r2 = float(r2_score(y_true_arr, y_pred))
    return {"rmse": rmse, "rmsle": rmsle, "r2": r2}


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Day 15's pipeline_factory.py with Day 15's own config...")
    pipeline_factory = _load_module_with_local_config(
        "pipeline_factory_day15",
        config.DAY15_PIPELINE_FACTORY_PATH,
        config.DAY15_CONFIG_PATH,
    )

    print(f"Loading data from {config.AMES_CSV_PATH}")
    df = load_raw_data()
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]  # kept as a Series — see module docstring

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    base_model = XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
    pipeline = pipeline_factory.build_pipeline(base_model)

    print(f"Running GridSearchCV: {config.PARAM_GRID} ({config.CV_FOLDS}-fold CV)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=config.PARAM_GRID,
        scoring=config.SCORING,
        cv=config.CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best params rediscovered: {best_params}")
    print(f"Best CV score (neg RMSE): {grid_search.best_score_:.2f}")

    y_pred = best_pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["rmse_ci"] = bootstrap_rmse_ci(y_test, y_pred)
    metrics["cv_best_score_neg_rmse"] = float(grid_search.best_score_)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))

    print(f"Test metrics: {metrics}")

    # Persist everything 02-05 need so nothing has to be refit.
    joblib.dump(best_pipeline, config.BEST_PIPELINE_PATH)
    joblib.dump((X_test, y_test), config.TEST_SPLIT_PATH)
    with open(config.BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    with open(config.BASELINE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved pipeline to {config.BEST_PIPELINE_PATH}")
    print(f"Saved test split to {config.TEST_SPLIT_PATH}")
    print(f"Saved baseline metrics to {config.BASELINE_METRICS_PATH}")


if __name__ == "__main__":
    main()
