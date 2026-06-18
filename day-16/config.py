# config.py — Day 16: Hyperparameter Tuning with Optuna + MLflow
# All constants, paths, and search space definitions live here.
# Nothing is hardcoded elsewhere.

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MLFLOW_DIR = BASE_DIR / "mlruns"

TRAIN_CSV = DATA_DIR / "train.csv"

# ── Experiment Identity ───────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "day16-xgboost-optuna-ames"
RANDOM_SEARCH_RUN_NAME = "random-search-10trials"
BAYESIAN_SEARCH_RUN_NAME = "bayesian-search-10trials"

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── Data ──────────────────────────────────────────────────────────────────────

TARGET_COL = "SalePrice"
TEST_SIZE = 0.2

# ── Cross-Validation ──────────────────────────────────────────────────────────

CV_FOLDS = 5
CV_SCORING = "neg_root_mean_squared_error"  # sklearn sign convention

# ── Optuna: Full Study (Standard Task) ───────────────────────────────────────

OPTUNA_N_TRIALS = 55  # 55 gives comfortable buffer over the 50-trial requirement
OPTUNA_STUDY_NAME = "ames-xgb-full"
OPTUNA_DIRECTION = "minimize"  # we minimize RMSE

# ── Optuna: Comparison Study (Out-of-Box Challenge) ───────────────────────────

COMPARISON_N_TRIALS = 50  # 10 random vs 10 Bayesian from the same starting point

# ── XGBoost Search Space ─────────────────────────────────────────────────────
# Used by both the full Optuna study and the Bayesian arm of the comparison.

XGB_SEARCH_SPACE = {
    "n_estimators": (100, 1000),  # (low, high) int range
    "max_depth": (3, 10),
    "learning_rate": (0.005, 0.3),  # float, log-scale sampled
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 10),
    "reg_alpha": (1e-8, 10.0),  # float, log-scale
    "reg_lambda": (1e-8, 10.0),  # float, log-scale
    "gamma": (0.0, 5.0),
}

# ── XGBoost Fixed Params ──────────────────────────────────────────────────────

XGB_FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Plot Aesthetics ───────────────────────────────────────────────────────────

PLOT_STYLE = "whitegrid"
PLOT_PALETTE = "muted"
FIGURE_DPI = 150
