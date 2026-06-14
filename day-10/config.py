"""
config.py — Day 10 shared constants.
All three scripts import from here so paths and hyperparameters
stay in one place.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW = (
    "/Users/sanjayachaudhary/Desktop/projects/day-10/data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv.xls"
)
OUTPUTS_DIR = "/Users/sanjayachaudhary/Desktop/projects/day-10/outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Model hyperparameters ─────────────────────────────────────────────────────
DT_MAX_DEPTH = 4
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
CV_N_SPLITS = 5

# ── Feature column groups (used by both 01_ and 02_) ─────────────────────────
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

YES_NO_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

MULTI_COLS = ["InternetService", "Contract", "PaymentMethod"]

TARGET_COL = "Churn"
