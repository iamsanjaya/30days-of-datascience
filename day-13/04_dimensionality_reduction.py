# PCA + t-SNE visualization of cluster structure
# Standard task: see if clusters are separable in 2D reduced space.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import (
    PROCESSED_DIR,
    PLOTS_DIR,
    PCA_N_COMPONENTS,
    TSNE_N_COMPONENTS,
    TSNE_PERPLEXITY,
    TSNE_N_ITER,
    RANDOM_STATE,
    FIGURE_DPI,
)

# %% Load (run-safe)
X_scaled = pd.read_csv(PROCESSED_DIR / "X_scaled.csv").values
df_clustered = pd.read_csv(PROCESSED_DIR / "df_clustered.csv")

labels = df_clustered["Cluster"].to_numpy(dtype=int)
n_clusters = len(np.unique(labels))

# =========================
# PCA
# =========================

# %% Fit PCA
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print(
    f"PCA explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f} "
    f"(total={explained.sum():.3f})"
)

# %% Scree + PCA scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA: Variance Capture and Cluster Structure", fontsize=13)

# Scree plot
all_pca = PCA(random_state=RANDOM_STATE).fit(X_scaled)
axes[0].bar(
    range(1, len(all_pca.explained_variance_ratio_) + 1),
    all_pca.explained_variance_ratio_,
    alpha=0.8,
)
axes[0].plot(
    range(1, len(all_pca.explained_variance_ratio_) + 1),
    np.cumsum(all_pca.explained_variance_ratio_),
    marker="o",
    label="Cumulative",
)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot")
axes[0].legend()

# PCA cluster scatter
for c in range(n_clusters):
    mask = labels == c
    axes[1].scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {c}",
        alpha=0.7,
        s=50,
    )

axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
axes[1].set_title("PCA Projection of Clusters")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "pca_visualization.png", dpi=FIGURE_DPI)
plt.close()

print("Saved: pca_visualization.png")

# =========================
# t-SNE
# =========================

print(f"\nFitting t-SNE (perplexity={TSNE_PERPLEXITY}, n_iter={TSNE_N_ITER}) ...")

tsne = TSNE(
    n_components=TSNE_N_COMPONENTS,
    perplexity=TSNE_PERPLEXITY,
    max_iter=TSNE_N_ITER,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE,
)

X_tsne = tsne.fit_transform(X_scaled)

# %% t-SNE scatter
fig, ax = plt.subplots(figsize=(8, 6))

for c in range(n_clusters):
    mask = labels == c
    ax.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        label=f"Cluster {c}",
        alpha=0.75,
        s=55,
    )

ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE Projection of Clusters")

ax.legend(fontsize=8)
ax.text(
    0.02,
    0.98,
    "t-SNE preserves local structure only",
    transform=ax.transAxes,
    fontsize=7,
    va="top",
    color="gray",
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "tsne_visualization.png", dpi=FIGURE_DPI)
plt.close()

print("Saved: tsne_visualization.png")

# =========================
# Save embeddings
# =========================

coords = pd.DataFrame(
    {
        "PCA_1": X_pca[:, 0],
        "PCA_2": X_pca[:, 1],
        "tSNE_1": X_tsne[:, 0],
        "tSNE_2": X_tsne[:, 1],
        "Cluster": labels,
    }
)

coords.to_csv(PROCESSED_DIR / "reduced_coords.csv", index=False)

print("Saved: reduced_coords.csv")
