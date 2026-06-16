from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "train.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "engineered.csv"
ABLATION_RESULTS_PATH = PROCESSED_DATA_DIR / "ablation_log.csv"

TARGET_COL = "SalePrice"

RANDOM_STATE = 42
CV_FOLDS = 5

DROP_COLS = [
    "Order",
    "PID",
    "Pool QC",
    "Misc Val",
    "Pool Area",
    "3Ssn Porch",
    "Low Qual Fin SF",
]

LOG_TRANSFORM_COLS = [
    "Lot Area",
    "Gr Liv Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "Garage Area",
    "Mas Vnr Area",
]

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "rmse",
}

ABLATION_RETENTION_THRESHOLD = 0.95


# ensure folders exist at import time
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
