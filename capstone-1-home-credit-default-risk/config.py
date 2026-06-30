"""
Config for Capstone 1 — Home Credit Default Risk.

Environment-aware path resolution: detects whether we're running locally
(M4 Mac, data inspection/EDA/feature engineering) or on a Kaggle notebook
(GPU training, Optuna tuning). One unified script set — no separate
local/Kaggle codebases.
"""

import os
from pathlib import Path

# --- Environment detection -------------------------------------------------


def _detect_kaggle() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    if Path("/kaggle/input").exists():
        return True
    return False


IS_KAGGLE: bool = _detect_kaggle()

# --- Dataset identity --------------------------------------------------------

# This is a *competition* dataset on Kaggle, not a generic dataset upload,
# so when added to a Kaggle notebook it mounts under this slug.
KAGGLE_DATASET_SLUG: str = os.environ.get(
    "KAGGLE_DATASET_SLUG", "home-credit-default-risk"
)

# --- Path resolution ---------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).parent

if IS_KAGGLE:
    RAW_DATA_DIR: Path = Path("/kaggle/input/competitions") / KAGGLE_DATASET_SLUG
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")
    MODEL_DIR: Path = Path("/kaggle/working/models")
else:
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    MODEL_DIR = PROJECT_ROOT / "models"

for _dir in (OUTPUT_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Raw file paths -----------------------------------------------------------

APPLICATION_TRAIN_PATH: Path = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH: Path = RAW_DATA_DIR / "application_test.csv"
BUREAU_PATH: Path = RAW_DATA_DIR / "bureau.csv"  # stretch goal only
COLUMNS_DESCRIPTION_PATH: Path = RAW_DATA_DIR / "HomeCredit_columns_description.csv"

# --- Modeling constants -------------------------------------------------------

TARGET_COL: str = "TARGET"
ID_COL: str = "SK_ID_CURR"
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2  # held-out test split, stratified on TARGET
N_CV_FOLDS: int = 5
PRIMARY_METRIC: str = "roc_auc"

# Sentinel value Home Credit uses for "not applicable" in DAYS_EMPLOYED
# (appears for pensioners/unemployed applicants) — must be treated as
# missing, not a real value of ~1000 years employed.
DAYS_EMPLOYED_ANOMALY_VALUE: int = 365243

# Optuna tuning budget (run on Kaggle, not locally — same pattern as Day 23+)
OPTUNA_N_TRIALS: int = 50

# Fewer folds during the Optuna search itself than the final 5-fold
# evaluation — 50 trials x 5 folds x LightGBM is expensive for marginal
# extra precision during search. Final best params still get evaluated
# with the full N_CV_FOLDS for the reported result.
OPTUNA_CV_FOLDS: int = 3

# --- MLflow ---------------------------------------------------------------

if IS_KAGGLE:
    MLFLOW_DB_PATH: Path = Path("/kaggle/working/mlflow.db")
else:
    MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

MLFLOW_TRACKING_URI: str = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT_NAME: str = "capstone-1-home-credit-default-risk"

# --- Optuna persistent storage --------------------------------------------
# Without this, a killed/disconnected Kaggle session loses all completed
# trials (the study object only lived in memory). With persistent storage,
# re-running the script resumes from the last completed trial instead of
# starting over from trial 0.

if IS_KAGGLE:
    OPTUNA_DB_PATH: Path = Path("/kaggle/working/optuna.db")
else:
    OPTUNA_DB_PATH = PROJECT_ROOT / "optuna.db"

OPTUNA_STORAGE_URI: str = f"sqlite:///{OPTUNA_DB_PATH}"
OPTUNA_STUDY_NAME: str = "capstone1_lightgbm_tuning"
