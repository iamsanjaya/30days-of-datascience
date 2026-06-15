# Fit final K-Means, profile clusters, save labels
# Standard task: who are these customers?

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

from config import (
    PROCESSED_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    CLUSTERING_FEATURES,
    KMEANS_MAX_ITER,
    KMEANS_N_INIT,
    RANDOM_STATE,
    FIGURE_DPI,
    PALETTE,
    REPORTS_DIR,
)

# %% Load (run-safe files)
X_scaled = pd.read_csv(PROCESSED_DIR / "X_scaled.csv").values
df = pd.read_csv(PROCESSED_DIR / "df_with_encoded.csv")

# ── Change K here if your elbow/silhouette suggests a different value ──────────
FINAL_K = 5

# %% Fit final K-Means
km = KMeans(
    n_clusters=FINAL_K,
    max_iter=KMEANS_MAX_ITER,
    n_init=KMEANS_N_INIT,
    random_state=RANDOM_STATE,
)

df["Cluster"] = km.fit_predict(X_scaled)
print(f"Cluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# %% Profile: mean of raw features per cluster
profile_cols = CLUSTERING_FEATURES + ["Gender_encoded"]
profile = df.groupby("Cluster")[profile_cols].mean().round(2)
profile["Size"] = df["Cluster"].value_counts().sort_index()
profile["% Female"] = (profile["Gender_encoded"] * 100).round(1)
profile = profile.drop(columns=["Gender_encoded"])

print("\nCluster Profiles (raw feature means):")
print(profile.to_string())

# %% Save labeled dataframe and profiles (run-safe)
df.to_csv(PROCESSED_DIR / "df_clustered.csv", index=False)
profile.to_csv(REPORTS_DIR / "cluster_profiles.csv", index=False)

print("\nSaved: df_clustered, cluster_profiles")

# %% Plot: Income vs Spending Score colored by cluster
cmap = plt.get_cmap(PALETTE)

fig, ax = plt.subplots(figsize=(9, 6))

for cluster_id in range(FINAL_K):
    mask = df["Cluster"] == cluster_id
    ax.scatter(
        df.loc[mask, "Annual Income (k$)"],
        df.loc[mask, "Spending Score (1-100)"],
        label=f"Cluster {cluster_id} (n={mask.sum()})",
        alpha=0.75,
        s=60,
    )

# Centroids (scaled → original space)
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
centers_raw = scaler.inverse_transform(km.cluster_centers_)
centers_raw_df = pd.DataFrame(centers_raw, columns=CLUSTERING_FEATURES)

ax.scatter(
    centers_raw_df["Annual Income (k$)"],
    centers_raw_df["Spending Score (1-100)"],
    marker="X",
    s=200,
    c="black",
    zorder=5,
    label="Centroids",
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title(f"K={FINAL_K} Clusters: Customer Segmentation")

ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "cluster_scatter.png", dpi=FIGURE_DPI)
plt.close()

print("Saved: cluster_scatter.png")
