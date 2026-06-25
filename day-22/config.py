"""
config.py — Day 22: CNN Data Efficiency

Single source of truth for paths and hyperparameters.
No magic numbers anywhere else in this day's scripts.

⚠️ ACTION REQUIRED BEFORE RUNNING:
1. DAY21_DATA_DIR  — point this at your Day 21 cleaned Cats vs Dogs splits
   (the folder produced by Day 21's data validation/prep script that contains
   train/val/test subfolders, each with cat/ and dog/ class folders).
2. NICHE_RAW_DATA_DIR — after downloading the "Devanagari Handwritten Character
   Dataset" from Kaggle, point this at the extracted root folder.

Assumptions documented inline below — adjust if your actual folder names differ.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

# --- Standard task: reuse Day 21's already-cleaned Cats vs Dogs splits ---
DAY21_ROOT = Path("/Users/sanjayachaudhary/Desktop/projects/day-21")
DAY21_SPLITS_DIR = DAY21_ROOT / "outputs" / "splits"
DAY21_TRAIN_CSV = DAY21_SPLITS_DIR / "train.csv"
DAY21_VAL_CSV = DAY21_SPLITS_DIR / "val.csv"
DAY21_TEST_CSV = DAY21_SPLITS_DIR / "test.csv"

# --- Niche task: Devanagari Handwritten Character Dataset (Kaggle) ---
# Download: "Devanagari Handwritten Character Dataset" (Kaggle).
# Assumption (UCI-derived structure, common to this dataset): after extraction,
# <NICHE_RAW_DATA_DIR>/Train/digit_0 .. digit_9/*.png
# <NICHE_RAW_DATA_DIR>/Test/digit_0 .. digit_9/*.png
# If your download unpacks with different subfolder names (e.g. capitalization,
# or a single combined folder instead of Train/Test), update NICHE_TRAIN_SUBDIR /
# NICHE_TEST_SUBDIR below — everything else still works off the class names.
NICHE_RAW_DATA_DIR = (
    PROJECT_ROOT / "data" / "raw" / "DevanagariHandwrittenCharacterDataset"
)
NICHE_TRAIN_SUBDIR = "Train"
NICHE_TEST_SUBDIR = "Test"
NICHE_CLASSES = [f"digit_{i}" for i in range(10)]  # only the 10 digit classes used

# --- Output locations (gitignored at runtime) ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBSETS_DIR = OUTPUTS_DIR / "subsets"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
NICHE_MANIFESTS_DIR = OUTPUTS_DIR / "niche_manifests"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_EFFICIENCY_RESULTS_PATH = RESULTS_DIR / "data_efficiency_results.json"
AUGMENTATION_RESULTS_PATH = RESULTS_DIR / "augmentation_results.json"
NICHE_RESULTS_PATH = RESULTS_DIR / "niche_results.json"

# ──────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────────────────────────────────
# STANDARD TASK — Cats vs Dogs data efficiency study
# ──────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)  # matches Day 21's ResNet50 input size
BATCH_SIZE = 32
SUBSET_FRACTIONS = [1.0, 0.5, 0.25, 0.10, 0.05]  # 100%, 50%, 25%, 10%, 5%

FROZEN_BACKBONE_LR = 1e-3
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

# Augmentation params used only in 03_augmentation_experiment.py
AUG_ROTATION = 0.08  # fraction of 2*pi
AUG_ZOOM = 0.15
AUG_CONTRAST = 0.10
# Which subset fraction(s) to re-train with augmentation and compare against
# the no-augmentation result already recorded for the same fraction.
AUGMENTATION_TARGET_FRACTIONS = [0.10, 0.05]

# ──────────────────────────────────────────────────────────────────────────
# OUT-OF-BOX CHALLENGE — Nepali Devanagari digit classifier, <500 images
# ──────────────────────────────────────────────────────────────────────────
NICHE_IMG_SIZE = (96, 96)  # upscaled from native 32x32 for transfer learning
NICHE_NUM_CLASSES = len(NICHE_CLASSES)

# The deliberately tiny, constrained training pool (the whole point of the
# challenge): 45 images/class x 10 classes = 450 images, under the 500 cap.
NICHE_TRAIN_PER_CLASS = 45

# Held-out evaluation sets — sampled from the *remaining* pool, never seen
# during training. These do not count toward the "<500 training images" cap;
# they exist purely to measure generalization fairly.
NICHE_VAL_PER_CLASS = 15
NICHE_TEST_PER_CLASS = 15

# Simulated "unlabeled" pool for the pseudo-labeling trick (06/10). These are
# real images from the same dataset with labels deliberately withheld from
# the model, to mimic having unlabeled data available in the wild. True
# labels are kept alongside for honest pseudo-label accuracy reporting —
# we are not fabricating any data, only withholding labels during training.
NICHE_UNLABELED_POOL_PER_CLASS = 150
PSEUDO_LABEL_CONFIDENCE_THRESHOLD = 0.90

NICHE_BATCH_SIZE = 16
NICHE_EPOCHS = 30
NICHE_FINE_TUNE_EPOCHS = 10
NICHE_HEAD_LR = 1e-3
NICHE_FINE_TUNE_LR = 1e-5
NICHE_EARLY_STOPPING_PATIENCE = 5

# Test-time augmentation (09_evaluate_with_tta.py)
TTA_NUM_AUGMENTATIONS = 8
