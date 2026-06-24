"""
Day 21 — CNNs: Transfer Learning + Grad-CAM Interpretability
All hyperparameters, paths, and constants live here. No magic numbers
in the pipeline scripts.

Backbone is configurable: set BACKBONE to "resnet50" or "xception".
Everything downstream (image size, preprocessing, Grad-CAM layer name,
model/output file paths) derives from that one setting.
"""

from pathlib import Path

# ── BACKBONE SELECTION ──────────────────────────────────────────────────
# Change this and rerun 02 -> 05 to train/evaluate the other backbone.
# Each backbone writes to its own model/output paths, so results from
# both are kept side by side rather than overwritten.
BACKBONE = "xception"  # "resnet50" | "xception"

# Per-backbone settings: native input size and the layer name to hook
# Grad-CAM into (the final conv activation before global pooling).
# Layer names confirmed via base_model.summary() — not assumed.
BACKBONE_SETTINGS = {
    "resnet50": {
        "img_size": (224, 224),
        "last_conv_layer_name": "conv5_block3_out",
    },
    "xception": {
        "img_size": (299, 299),
        "last_conv_layer_name": "block14_sepconv2_act",
    },
}

# ── PATHS ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

# Microsoft Cats vs Dogs dataset (kaggle.com/datasets/shaunthesheep/
# microsoft-catsvsdogs-dataset). Expected after extraction:
#   data/raw/PetImages/Cat/0.jpg, 1.jpg, ...
#   data/raw/PetImages/Dog/0.jpg, 1.jpg, ...
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "PetImages"
CAT_DIR_NAME = "Cat"
DOG_DIR_NAME = "Dog"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUT_DIR / "splits"
CURVES_DIR = OUTPUT_DIR / "curves" / BACKBONE
GRADCAM_DIR = OUTPUT_DIR / "gradcam" / BACKBONE
COMPARISON_DIR = OUTPUT_DIR / "comparison" / BACKBONE
DATA_QUALITY_DIR = OUTPUT_DIR / "data_quality"
MODEL_DIR = PROJECT_ROOT / "models"

FROZEN_MODEL_PATH = MODEL_DIR / f"frozen_model_{BACKBONE}.keras"
FINETUNED_MODEL_PATH = MODEL_DIR / f"finetuned_model_{BACKBONE}.keras"

FROZEN_HISTORY_PATH = OUTPUT_DIR / f"frozen_history_{BACKBONE}.json"
FINETUNE_HISTORY_PATH = OUTPUT_DIR / f"finetune_history_{BACKBONE}.json"

# Splits are shared across backbones — same train/val/test images,
# regardless of which model is being trained on them.
TRAIN_SPLIT_CSV = SPLITS_DIR / "train.csv"
VAL_SPLIT_CSV = SPLITS_DIR / "val.csv"
TEST_SPLIT_CSV = SPLITS_DIR / "test.csv"

CORRUPT_FILES_LOG = DATA_QUALITY_DIR / "corrupt_files_removed.csv"

# ── REPRODUCIBILITY ───────────────────────────────────────────────────
SEED = 42

# ── DATA ───────────────────────────────────────────────────────────────
IMG_SIZE = BACKBONE_SETTINGS[BACKBONE]["img_size"]
CHANNELS = 3
BATCH_SIZE = 32

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Set to an int (e.g. 4000) to subsample the ~25k dataset for faster
# iteration on M4 while developing. None = use the full dataset.
SAMPLE_SIZE = None

# ── MODEL ──────────────────────────────────────────────────────────────
NUM_CLASSES = 1  # binary: sigmoid output (0 = cat, 1 = dog)
DROPOUT_RATE = 0.3
DENSE_UNITS = 256

LAST_CONV_LAYER_NAME = BACKBONE_SETTINGS[BACKBONE]["last_conv_layer_name"]

# Number of trailing layers of the backbone to unfreeze during Phase 2
# (fine-tuning), counted from the end of the base model.
UNFREEZE_LAYERS = 10

# ── TRAINING — PHASE 1: FROZEN BASE ────────────────────────────────────
FROZEN_EPOCHS = 10
FROZEN_LR = 1e-3

# ── TRAINING — PHASE 2: FINE-TUNE ──────────────────────────────────────
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-5  # must be small — we're nudging pretrained weights

# ── CALLBACKS ────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5

# ── GRAD-CAM ANALYSIS (out-of-box challenge) ───────────────────────────
NUM_CORRECT_SAMPLES = 5
NUM_MISCLASSIFIED_SAMPLES = 5
GRADCAM_ALPHA = 0.4  # heatmap overlay opacity

# ── NOTES ───────────────────────────────────────────────────────────────
# tensorflow-metal / Apple M4 note: BatchNorm layers (present throughout
# both ResNet50 and Xception) have historically had compatibility issues
# with the default tf.keras.optimizers.Adam under tensorflow-metal.
# training.compile_model() uses tf.keras.optimizers.legacy.Adam,
# consistent with the Day 19/20 fix already in this environment.
#
# Data quality note: the Microsoft Cats vs Dogs dataset contains a small
# number of zero-byte / undecodable files (confirmed: Cat/666.jpg,
# Dog/11702.jpg — and there may be others). 01_prepare_data.py validates
# every file before it ever reaches training, rather than relying on
# the data loader to skip bad files at read time (it won't — see
# utils/data.py docstring for why).
