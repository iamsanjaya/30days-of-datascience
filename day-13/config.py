# Day 13: Clustering & Dimensionality Reduction
# Mall Customers dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

MODELS_DIR = BASE_DIR / "models"

DATA_FILE = RAW_DIR / "Mall_Customers.csv"

for p in [PROCESSED_DIR, PLOTS_DIR, REPORTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# Feature config
CLUSTERING_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

# K-Means
KMEANS_K_RANGE = range(2, 11)
KMEANS_K_VALUES = [3, 5, 8]
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10

# DBSCAN
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# PCA
PCA_N_COMPONENTS = 2

# t-SNE
TSNE_N_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# Plot style
FIGURE_DPI = 150
PALETTE = "tab10"
