import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Breast_Cancer.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Dataset
TARGET_COL = "Status"
POSITIVE_CLASS = "Dead"
DROP_COLUMNS = []

# Experiment
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model
MODEL_N_ESTIMATORS = 100
MODEL_MAX_DEPTH = 5

# Imbalance Ratios (Out-of-Box Challenge)
IMBALANCE_RATIOS = [
    ("1:2", 0.50),
    ("1:5", 0.20),
    ("1:10", 0.10),
    ("1:50", 0.02),
    ("1:100", 0.01),
]

# Plot Colours
COLORS = {
    "baseline": "#C44E52",
    "smote": "#4C72B0",
    "undersampling": "#55A868",
    "classweight": "#DD8452",
}
