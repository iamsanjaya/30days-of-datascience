# Elbow method + Silhouette score to select optimal K
# Standard task: find the right K before committing to a cluster count.

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    PROCESSED_DIR,
    KMEANS_K_RANGE,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    RANDOM_STATE,
    FIGURE_DPI,
    PLOTS_DIR,
)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
# %% Load scaled features (run-safe file)
X_scaled = pd.read_csv(PROCESSED_DIR / "X_scaled.csv").values

# %% Compute inertia (elbow) + silhouette for each K
inertias = []
silhouette_scores = []
k_values = list(KMEANS_K_RANGE)

for k in k_values:
    km = KMeans(
        n_clusters=k,
        max_iter=KMEANS_MAX_ITER,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels) if k > 1 else None
    silhouette_scores.append(sil)
    print(f"K={k:2d} | Inertia={km.inertia_:8.1f} | Silhouette={sil:.4f}")

# %% Plot: Elbow + Silhouette side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("K Selection: Elbow Method and Silhouette Score", fontsize=14)

# Elbow
axes[0].plot(k_values, inertias, marker="o", linewidth=2)
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-Cluster SSE)")
axes[0].set_title("Elbow Method — Look for the Bend")
axes[0].set_xticks(k_values)

# Silhouette
sil_values = [s for s in silhouette_scores if s is not None]
sil_k = [k for k, s in zip(k_values, silhouette_scores) if s is not None]
axes[1].plot(sil_k, sil_values, marker="s", linewidth=2)
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score (higher = better)")
axes[1].set_title("Silhouette Score — Higher is More Cohesive")
axes[1].set_xticks(sil_k)

# Annotate best silhouette
best_k = sil_k[int(np.argmax(sil_values))]
best_sil = max(sil_values)
axes[1].annotate(
    f"Best K={best_k}\n({best_sil:.3f})",
    xy=(best_k, best_sil),
    xytext=(best_k + 0.5, best_sil - 0.02),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=9,
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "k_selection.png", dpi=FIGURE_DPI)
plt.close()

print(f"\nBest K by silhouette: {best_k} (score={best_sil:.4f})")
print("Note: Elbow and silhouette may disagree — that is expected and informative.")
print("Saved: k_selection.png")
