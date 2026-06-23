"""Day 20 — Configuration: paths, hyperparameters, and experiment settings.

Everything that varies between runs lives here, not scattered across scripts.
"""

import os

# Must be set before TensorFlow is imported anywhere (here or in any script/
# util that imports config first). Suppresses the native C++/Metal-plugin
# info logs — e.g. "stateless_random_op... GPU implementation does not
# produce the same series as CPU" printed once per augmentation call — which
# otherwise flood the terminal during augmented training runs. This is a
# logging-verbosity setting only; it does not affect training behavior.
# Levels: "0" all logs, "1" hide INFO, "2" hide INFO+WARNING, "3" hide all but errors.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DAY_ROOT = Path(__file__).parent
DATA_DIR = DAY_ROOT / "data" / "raw" / "cifar10"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUTS_DIR = DAY_ROOT / "outputs"
HISTORIES_DIR = OUTPUTS_DIR / "histories"
MODELS_DIR = DAY_ROOT / "models"

for directory in (OUTPUTS_DIR, HISTORIES_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = 32
NUM_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

# Must match the folder names in the Kaggle "CIFAR-10 PNGs in folders" dataset.
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
NUM_CLASSES = len(CLASS_NAMES)

RANDOM_SEED = 42

# ── Deliberate overfit setup ─────────────────────────────────────────────────
# Small data + large model = textbook overfitting, on purpose.
OVERFIT_SAMPLES_PER_CLASS = 50  # 500 training images total
OVERFIT_VAL_SAMPLES_PER_CLASS = 20  # 200 validation images total
OVERFIT_EPOCHS = 100
BASELINE_BATCH_SIZE = 32

# This architecture has no BatchNorm anywhere (that's a fix tested separately
# in 03), so weight magnitudes can drift over 100 epochs and Adam's adaptive
# scaling compounds rather than damps that drift. 1e-3 diverges into
# numerical blowup well before it overfits. 1e-4 + gradient-norm clipping
# keeps training numerically stable WITHOUT capping model capacity — this is
# a training-stability fix, not a regularization fix, so it doesn't quietly
# borrow from what 03 is supposed to be testing.
BASELINE_LEARNING_RATE = 1e-4
BASELINE_CLIPNORM = 1.0

# ── Regularization fixes ─────────────────────────────────────────────────────
DROPOUT_RATE = 0.5
L2_LAMBDA = 1e-3
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MONITOR = "val_loss"

AUGMENTATION_ROTATION = 0.08  # fraction of 2*pi radians
AUGMENTATION_ZOOM = 0.15
AUGMENTATION_TRANSLATION = 0.1

# One entry per model trained in 03_regularization_comparison.py.
# "baseline" is not retrained there — its history is reused from 02.
REGULARIZATION_CONFIGS = {
    "baseline": {
        "dropout": 0.0,
        "l2": 0.0,
        "batchnorm": False,
        "augment": False,
        "early_stop": False,
    },
    "dropout": {
        "dropout": DROPOUT_RATE,
        "l2": 0.0,
        "batchnorm": False,
        "augment": False,
        "early_stop": False,
    },
    "l2": {
        "dropout": 0.0,
        "l2": L2_LAMBDA,
        "batchnorm": False,
        "augment": False,
        "early_stop": False,
    },
    "batchnorm": {
        "dropout": 0.0,
        "l2": 0.0,
        "batchnorm": True,
        "augment": False,
        "early_stop": False,
    },
    "augmentation": {
        "dropout": 0.0,
        "l2": 0.0,
        "batchnorm": False,
        "augment": True,
        "early_stop": False,
    },
    "early_stopping": {
        "dropout": 0.0,
        "l2": 0.0,
        "batchnorm": False,
        "augment": False,
        "early_stop": True,
    },
    "combined": {
        "dropout": DROPOUT_RATE,
        "l2": L2_LAMBDA,
        "batchnorm": True,
        "augment": True,
        "early_stop": True,
    },
}

# ── LR Range Test (out-of-box challenge) ─────────────────────────────────────
LR_RANGE_START = 1e-7
LR_RANGE_END = 10.0
LR_RANGE_BATCH_SIZE = 128
LR_SMOOTHING_BETA = 0.98  # EMA smoothing factor for the noisy per-batch loss
LR_DIVERGE_FACTOR = 4.0  # flag divergence once smoothed loss exceeds this
# multiple of the running minimum
