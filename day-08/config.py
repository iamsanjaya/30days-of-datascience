"""
Day 08: Linear Regression
Single source of truth for all constants.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "housing.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_SAMPLES = 300
LEARNING_RATE = 0.01
N_ITERATIONS = 1000
FIGSIZE = (10, 6)
DPI = 300
