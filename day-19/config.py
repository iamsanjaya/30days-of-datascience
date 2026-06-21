"""
Central configuration for Day 19 — Keras Architecture Search + Lottery Ticket Pruning.
All paths, hyperparameters, and grid definitions live here. No magic numbers in scripts.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "adult.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = DATA_DIR / "processed"
ARCH_SEARCH_DIR = OUTPUT_DIR / "architecture_search"
BEST_MODEL_OUTPUT_DIR = OUTPUT_DIR / "best_model"
PRUNING_OUTPUT_DIR = OUTPUT_DIR / "pruning"

MODELS_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
BEST_MODEL_INIT_WEIGHTS_PATH = MODELS_DIR / "best_model_initial_weights.npz"
PRUNED_MODEL_PATH = MODELS_DIR / "pruned_model.keras"
RANDOM_PRUNED_MODEL_PATH = MODELS_DIR / "random_pruned_model.keras"

for d in [
    PROCESSED_DIR,
    ARCH_SEARCH_DIR,
    BEST_MODEL_OUTPUT_DIR,
    PRUNING_OUTPUT_DIR,
    MODELS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# RAW DATA SCHEMA (Kaggle uciml/adult-census-income)
# ----------------------------------------------------------------------------
TARGET_COL = "income"
POSITIVE_LABEL = ">50K"  # mapped to 1

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

MISSING_TOKEN = "?"

# ----------------------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# ----------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the remaining (non-test) data

# ----------------------------------------------------------------------------
# ARCHITECTURE SEARCH GRID
# ----------------------------------------------------------------------------
DEPTHS = [1, 2, 3, 4, 5]  # number of hidden layers
WIDTHS = [32, 64, 128, 256, 512]  # neurons per hidden layer (uniform across depth)
DROPOUT_RATES = [0.0, 0.2, 0.5]

SEARCH_EPOCHS = 15
SEARCH_BATCH_SIZE = 256
SEARCH_PATIENCE = 3  # early stopping patience during search
SEARCH_LEARNING_RATE = 1e-3

GRID_RESULTS_CSV = ARCH_SEARCH_DIR / "grid_results.csv"

# ----------------------------------------------------------------------------
# FINAL (BEST ARCHITECTURE) TRAINING
# ----------------------------------------------------------------------------
FINAL_EPOCHS = 60
FINAL_BATCH_SIZE = 256
FINAL_PATIENCE = 8
FINAL_LEARNING_RATE = 1e-3

BEST_MODEL_SELECTION_METRIC = "val_accuracy"  # column in grid_results.csv to rank by

# ----------------------------------------------------------------------------
# LOTTERY TICKET PRUNING
# ----------------------------------------------------------------------------
PRUNING_FRACTION = 0.5  # fraction of smallest-magnitude weights to zero out
PRUNING_EPOCHS = 60
PRUNING_BATCH_SIZE = 256
PRUNING_PATIENCE = 8
PRUNING_LEARNING_RATE = 1e-3

PRUNING_COMPARISON_CSV = PRUNING_OUTPUT_DIR / "comparison.csv"
