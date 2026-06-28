"""
Day 23 — NLP: From Bag-of-Words to BERT
Central configuration. No magic numbers in any pipeline script — everything lives here.
"""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Environment detection ───────────────────────────────────────────────
# Kaggle kernels set KAGGLE_KERNEL_RUN_TYPE and mount input data read-only
# under /kaggle/input/. Locally (your M4 / project-env) we use a normal
# data/ folder next to this file. Everything below resolves from this flag
# so the same script runs unmodified in both places.
IS_KAGGLE = (
    bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/input").exists()
)

DAY_ROOT = Path(__file__).parent

if IS_KAGGLE:
    # This dataset (crowdflower/twitter-airline-sentiment) mounts under a
    # nested "organizations" path rather than the usual flat /kaggle/input/<slug>/
    # pattern — confirmed via os.walk("/kaggle/input") on 2026-06-28:
    #   /kaggle/input/datasets/organizations/crowdflower/twitter-airline-sentiment/Tweets.csv
    # Override via KAGGLE_DATASET_DIR env var if you ever re-attach it differently
    # (Notebook -> Add-ons -> Environment Variables) instead of editing this file.
    _DEFAULT_KAGGLE_DATASET_DIR = (
        "/kaggle/input/datasets/organizations/crowdflower/twitter-airline-sentiment"
    )
    RAW_DATA_DIR = Path(
        os.environ.get("KAGGLE_DATASET_DIR", _DEFAULT_KAGGLE_DATASET_DIR)
    )
    _WORKING_DIR = Path("/kaggle/working")
    PROCESSED_DATA_DIR = _WORKING_DIR / "processed"
    OUTPUT_DIR = _WORKING_DIR / "outputs"
    MODELS_DIR = _WORKING_DIR / "models"
else:
    DATA_DIR = DAY_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = DAY_ROOT / "outputs"
    MODELS_DIR = DAY_ROOT / "models"

# /kaggle/input is a read-only mount — only create the dirs that are actually writable.
_writable_dirs = (PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR)
if not IS_KAGGLE:
    _writable_dirs = (RAW_DATA_DIR,) + _writable_dirs
for _d in _writable_dirs:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────
# Kaggle source: crowdflower/twitter-airline-sentiment
# Locally: download "Tweets.csv", rename to "tweets.csv", place at data/raw/tweets.csv
# On Kaggle: the public dataset ships the file as "Tweets.csv" (capital T) —
# RAW_CSV_NAME below resolves the casing per-environment so RAW_CSV_PATH is
# correct in both places without touching the read-only Kaggle mount.
RAW_CSV_NAME = "Tweets.csv" if IS_KAGGLE else "tweets.csv"
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_NAME

TEXT_COLUMN = "text"
LABEL_COLUMN = "airline_sentiment"

# Raw label string -> integer class id. Order defines CLASS_NAMES index alignment.
LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
CLASS_NAMES = ["negative", "neutral", "positive"]
NUM_CLASSES = len(CLASS_NAMES)

# Processed split files (written by 01_data_preparation.py)
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# ── Reproducibility ──────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Split sizes ──────────────────────────────────────────────────────────
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the remaining (non-test) data

# ── Text cleaning ────────────────────────────────────────────────────────
# Two cleaning levels (utils/preprocessing.py):
#   light_clean      -> BERT (keeps casing/punctuation; subword tokenizer handles the rest)
#   aggressive_clean -> TF-IDF and Word2Vec+LSTM (lowercase, strip noise, remove stopwords)
MIN_TOKEN_LENGTH = 1

# ── TF-IDF + Logistic Regression baseline ───────────────────────────────
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0

# ── Word2Vec ─────────────────────────────────────────────────────────────
W2V_EMBEDDING_DIM = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 20
W2V_SG = 1  # 1 = skip-gram, 0 = CBOW

# ── LSTM (tf.keras) ──────────────────────────────────────────────────────
# No BatchNorm in this architecture -> standard tf.keras.optimizers.Adam is fine on Metal.
LSTM_MAX_VOCAB = 20_000
LSTM_MAX_LEN = 40
LSTM_UNITS = 64
LSTM_DROPOUT = 0.3
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 25
LSTM_LEARNING_RATE = 1e-3
LSTM_PATIENCE = 4  # EarlyStopping patience

# ── BERT (HuggingFace + PyTorch) ─────────────────────────────────────────
BERT_MODEL_NAME = (
    "distilbert-base-uncased"  # lighter than bert-base, fine for an M4 laptop
)
BERT_MAX_LEN = 64
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_WARMUP_RATIO = 0.1
# Apple Silicon: PyTorch's Metal backend is "mps". Kaggle notebooks get an NVIDIA
# GPU exposed as "cuda". training/bert_utils falls back down this list to "cpu".
BERT_DEVICE_PREFERENCE = ["cuda", "mps", "cpu"]

# ── Failure forensics (Day 23 out-of-box challenge) ─────────────────────
FORENSICS_NUM_FAILURES = 8  # pull more than 5 so the clearest cases can be chosen
FORENSICS_CONFIDENCE_THRESHOLD = (
    0.70  # "confidently wrong" cutoff on the predicted class prob
)
NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "n't",
    "cannot",
    "cant",
    "wont",
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
    "without",
}

# ── Misc ──────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
