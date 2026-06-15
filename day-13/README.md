# Day 13 — Clustering & Dimensionality Reduction

**Dataset:** Mall Customers — [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
**Download:** `Mall_Customers.csv` → place in `data/`

---

## What I built

Unsupervised customer segmentation pipeline using K-Means, PCA, and t-SNE on a real retail dataset. Standard task + an out-of-box challenge that proves clustering is subjective by design. The pipeline follows a single-run, non-versioned structure. All intermediate artifacts are overwritten per execution for simplicity and learning clarity.

### File structure

```
day-13/
├── config.py                       # all paths, constants, hyperparameters
├── 01_preprocess.py                # load, encode Gender, StandardScaler
├── 02_kmeans_selection.py          # elbow method + silhouette score → pick K
├── 03_cluster_profiles.py          # fit final K-Means, profile each cluster
├── 04_dimensionality_reduction.py  # PCA scree + t-SNE cluster scatter
├── 05_subjectivity.py              # K=3 vs K=5 vs K=8 — clustering as a lens
├── README.md
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv
│   └── processed/
│       ├── X_scaled.csv
│       ├── df_with_encoded.csv
│       ├── df_clustered.csv
│       └── reduced_coords.csv
│
├── models/
│   └── scaler.joblib
│
└── outputs/
    ├── plots/
    │   ├── k_selection.png
    │   ├── cluster_scatter.png
    │   ├── pca_visualization.png
    │   ├── tsne_visualization.png
    │   └── subjectivity_comparison.png
    │
    └── reports/
        ├── cluster_profiles.csv
        └── clustering_subjectivity_insight.txt

learning-journal/
  └── day-13.md

```

---

## Run order

```bash
python 01_preprocess.py
python 02_kmeans_selection.py
python 03_cluster_profiles.py
python 04_dimensionality_reduction.py
python 05_outofbox_subjectivity.py
```

---

## Standard task results

- **Elbow + silhouette** tested for K=2–10
- Silhouette peaks at K=5 (score ≈ 0.55); elbow bends around K=5 as well
- Final K=5 cluster profiles:

| Cluster | Avg Income | Avg Spend | Interpretation                            |
| ------- | ---------- | --------- | ----------------------------------------- |
| 0       | Low        | Low       | Budget necessity shoppers                 |
| 1       | Mid        | Mid       | Average household, occasional buyers      |
| 2       | High       | High      | Premium loyal customers — highest LTV     |
| 3       | Low        | High      | Young over-spenders — retention risk      |
| 4       | High       | Low       | Wealthy non-converters — hard to activate |

- PCA 2D and t-SNE 2D both confirm cluster separation
- Centroids plotted in original feature space with Income vs Spending Score scatter

---

## Out-of-Box Challenge — "Clustering is Subjective. Prove It."

Ran K-Means for **K=3, K=5, K=8** on the same data and wrote business descriptions for each:

| K   | Silhouette | Business story                                                       |
| --- | ---------- | -------------------------------------------------------------------- |
| 3   | ~0.43      | Budget / Average / Premium — 3-act story                             |
| 5   | ~0.55      | Adds young over-spenders + wealthy non-buyers as actionable segments |
| 8   | ~0.40      | Over-partitions; segments lose distinct meaning                      |

**Key insight:**
K=3's "premium" cluster is simply K=8's clusters 2+3 merged. They tell a consistent story at coarser resolution. K=8 creates false granularity — silhouette drops and centroids blur into noise. The "right" K is not a mathematical truth — it is the granularity that serves the business question being asked.

Full written insight saved to `outputs/clustering_subjectivity_insight.txt`.

---

## Key concepts

- **K-Means**: iterative centroid assignment; sensitive to scale → always `StandardScaler` first
- **Elbow method**: plot inertia vs K; bend = candidate K (often ambiguous)
- **Silhouette score**: [-1, 1]; higher = tighter, better-separated clusters; most reliable K selector
- **PCA**: linear projection; fast; good for scree plots and variance decomposition
- **t-SNE**: non-linear; preserves local structure; axes have no interpretable units; slow on large N
- **Clustering subjectivity**: K is a lens. The question determines the right granularity.
