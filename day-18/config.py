from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data" / "raw"
TRAIN_CSV: Path = DATA_DIR / "train.csv"

MODELS_DIR: Path = ROOT_DIR / "models"

OUTPUT_DIR: Path = ROOT_DIR / "outputs"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
METRICS_DIR: Path = OUTPUT_DIR / "metrics"

KERAS_LEARNING_RATE: float = 0.001
NUMPY_LEARNING_RATE: float = 0.001

KERAS_EPOCHS: int = 10
NUMPY_EPOCHS: int = 10

KERAS_BATCH_SIZE: int = 32
NUMPY_BATCH_SIZE: int = 32

INPUT_SIZE: int = 784
HIDDEN_SIZE: int = 128
OUTPUT_SIZE: int = 10
NUM_CLASSES: int = 10

TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42

RANDOM_LABEL_EPOCHS: int = 20
