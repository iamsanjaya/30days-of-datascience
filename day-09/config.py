from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "condition"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5

# Cost framework ( out - of - box challenge)
# Units: arbitrary "harm units" for decision document
FALSE_NEGATIVE_COST = 10  # missed disease - patient untreated
FALSE_POSITIVE_COST = 1  # unnecessary follow-up - time + stress
