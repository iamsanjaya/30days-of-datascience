# config.py — Day 15: sklearn Pipelines + Data Leakage Quiz
# Ames Housing Dataset (2930 rows, 82 cols)

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "AmesHousing.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Train / Test split ────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Target column ─────────────────────────────────────────────────────────────
TARGET = "SalePrice"

# ── Columns to drop before modelling ─────────────────────────────────────────
# Order and PID are row identifiers — no predictive signal
DROP_COLS = ["Order", "PID"]

# ── Numeric features (36) ────────────────────────────────────────────────────
# Garage Yr Blt is numeric but has 159 missing → median imputation
NUMERIC_FEATURES = [
    "MS SubClass",
    "Lot Frontage",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
]

# ── Categorical features (43) ─────────────────────────────────────────────────
# High-missing cats like Pool QC, Alley, Fence → NaN means "None/Absent",
# so most_frequent imputation is used; OHE handles unknowns at test time.
CATEGORICAL_FEATURES = [
    "MS Zoning",
    "Street",
    "Alley",
    "Lot Shape",
    "Land Contour",
    "Utilities",
    "Lot Config",
    "Land Slope",
    "Neighborhood",
    "Condition 1",
    "Condition 2",
    "Bldg Type",
    "House Style",
    "Roof Style",
    "Roof Matl",
    "Exterior 1st",
    "Exterior 2nd",
    "Mas Vnr Type",
    "Exter Qual",
    "Exter Cond",
    "Foundation",
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "Heating",
    "Heating QC",
    "Central Air",
    "Electrical",
    "Kitchen Qual",
    "Functional",
    "Fireplace Qu",
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond",
    "Paved Drive",
    "Pool QC",
    "Fence",
    "Misc Feature",
    "Sale Type",
    "Sale Condition",
]

# ── Numeric pipeline settings ─────────────────────────────────────────────────
NUMERIC_IMPUTE_STRATEGY = "median"  # robust to skew in SF / price columns
SCALER = "standard"  # StandardScaler — required for Ridge baseline

# ── Categorical pipeline settings ─────────────────────────────────────────────
CAT_IMPUTE_STRATEGY = "most_frequent"
OHE_HANDLE_UNKNOWN = "ignore"  # unseen categories at test time → all-zero row

# ── Model hyperparameters ─────────────────────────────────────────────────────
# XGBoost final estimator defaults (tuned in 03_grid_search.py)
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── GridSearchCV settings ─────────────────────────────────────────────────────
CV_FOLDS = 5
SCORING = "neg_root_mean_squared_error"

GRID_PARAM_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 4, 6],
    "model__learning_rate": [0.05, 0.10],
}
