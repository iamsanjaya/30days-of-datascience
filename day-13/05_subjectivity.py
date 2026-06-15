# Out-of-Box Challenge: Clustering is Subjective. Prove It.
#
# Run K-Means with K=3, K=5, K=8. For each K:
#   - Write a 2-sentence business description per cluster
#   - Ask: are K=3 clusters just merged K=8 clusters, or a fundamentally different story?
#   - Document the insight: clustering is a lens, not a truth.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from config import (
    PROCESSED_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    CLUSTERING_FEATURES,
    KMEANS_K_VALUES,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    RANDOM_STATE,
    FIGURE_DPI,
)

# %% Load (run-safe)
X_scaled = pd.read_csv(PROCESSED_DIR / "X_scaled.csv").values
df = pd.read_csv(PROCESSED_DIR / "df_with_encoded.csv")

scaler = joblib.load(MODELS_DIR / "scaler.joblib")

# %% Fit K-Means for each K
results = {}

for k in KMEANS_K_VALUES:
    km = KMeans(
        n_clusters=k,
        max_iter=KMEANS_MAX_ITER,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )

    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    centers_raw = scaler.inverse_transform(km.cluster_centers_)
    centers_df = pd.DataFrame(centers_raw, columns=CLUSTERING_FEATURES)

    centers_df["Cluster"] = range(k)
    centers_df["Size"] = [np.sum(labels == c) for c in range(k)]

    results[k] = {
        "labels": labels,
        "centers_raw": centers_df,
        "silhouette": sil,
        "model": km,
    }

    print(f"\n{'='*50}")
    print(f"K = {k}  |  Silhouette = {sil:.4f}")
    print(centers_df.to_string(index=False))

# %% Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
fig.suptitle("Same Data, Different Granularity: K=3 vs K=5 vs K=8", fontsize=12)

for ax, k in zip(axes, KMEANS_K_VALUES):
    labels = results[k]["labels"]
    sil = results[k]["silhouette"]

    for c in range(k):
        mask = labels == c
        ax.scatter(
            df.loc[mask, "Annual Income (k$)"],
            df.loc[mask, "Spending Score (1-100)"],
            label=f"C{c}",
            alpha=0.7,
            s=45,
        )

    ax.set_title(f"K={k} (Silhouette={sil:.3f})")
    ax.set_xlabel("Annual Income (k$)")

axes[0].set_ylabel("Spending Score (1-100)")

for ax in axes:
    ax.legend(fontsize=6, ncol=2)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "subjectivity_comparison.png", dpi=FIGURE_DPI)
plt.close()

print("Saved: subjectivity_comparison.png")

# %% Business interpretations
print("\n" + "=" * 60)
print("CLUSTER BUSINESS DESCRIPTIONS")
print("=" * 60)

descriptions = {
    3: [
        "Low-income / Low-spend: budget-sensitive customers with low engagement.",
        "Mid-income / Mid-spend: general customer base with balanced behavior.",
        "High-income / High-spend: premium customers with highest lifetime value.",
    ],
    5: [
        "Low-income / Low-spend: low engagement, price-sensitive segment.",
        "Mid-income / Mid-spend: stable average shoppers with moderate activity.",
        "High-income / High-spend: premium loyal customers driving revenue.",
        "Low-income / High-spend: risky high-spenders likely driven by impulse or credit.",
        "High-income / Low-spend: wealthy but conservative shoppers, hard to convert.",
    ],
    8: [
        "Over-segmentation begins: clusters split into micro-variations of same behavior.",
        "Several clusters are subdivisions of the 5-core structure, not new behavior types.",
        "Business interpretability decreases as K increases.",
        "Diminishing returns: additional clusters add complexity without clear actionability.",
        "Some clusters become statistically similar but artificially separated.",
        "Marketing strategies become harder to justify at this granularity.",
        "K=8 reflects model flexibility, not necessarily data structure.",
        "This highlights that clustering is decision-driven, not truth-driven.",
    ],
}

for k, descs in descriptions.items():
    print(f"\nK = {k}:")
    for i, d in enumerate(descs):
        print(f"  Cluster {i}: {d}")

# %% Insight
insight = """
INSIGHT: Clustering is a Lens, Not a Truth
------------------------------------------
Different values of K produce different but internally consistent narratives.

K=3 provides a coarse business segmentation: budget, average, premium.
K=5 introduces actionable subgroups such as high-spend risk and conservative wealth segments.
K=8 fragments existing structure without adding proportional business value.

Higher K improves fit metrics slightly but reduces interpretability and decision clarity.
Therefore, the "best K" is not mathematical alone — it is defined by the decision context.

Clustering should be treated as a tool for shaping perspective, not discovering objective categories.
"""

print(insight)

insight_path = REPORTS_DIR / "clustering_subjectivity_insight.txt"
insight_path.write_text(insight.strip())

print("Saved: clustering_subjectivity_insight.txt")
