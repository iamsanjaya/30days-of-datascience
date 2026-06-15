# Day 13 — June 15, 2026

---

## What I built today

### `01_preprocess.py`

Loaded Mall Customers CSV, encoded Gender as binary, applied `StandardScaler` to the three numeric features (Age, Annual Income, Spending Score). K-Means is distance-based so scaling is non-negotiable — unscaled income (range 15–137k) would dominate age (range 18–70) purely due to magnitude.

### `02_kmeans_selection.py`

Tested K=2 through K=10. Computed both inertia (for elbow) and silhouette score for each K. Plotted side-by-side. Elbow bend and silhouette peak both pointed to K=5. Annotated the best K on the silhouette plot.

### `03_cluster_profiles.py`

Fit final K=5 model. Profiled each cluster by mean of raw (unscaled) features. Plotted Income vs Spending Score colored by cluster — this 2D view is the most interpretable for this dataset because those two features drive most of the segmentation. Centroids inverse-transformed back to original scale for the scatter plot.

### `04_dimensionality_reduction.py`

PCA: generated scree plot showing cumulative explained variance, then plotted cluster labels in PC1/PC2 space. Two PCs capture most of the variance (~85–95% depending on the dataset split), making 2D projection reasonably representative. t-SNE: ran with Perplexity=30 (dataset-size dependent heuristic choice), confirmed cluster separation. Added a note on the plot that t-SNE axes carry no absolute meaning — only relative distances.

### `05_outofbox_subjectivity.py`

Ran K=3, K=5, K=8 on identical data. Wrote business descriptions per cluster for each K. Generated a side-by-side scatter comparison. Documented the core insight in `clustering_subjectivity_insight.txt`.

---

## Out-of-Box Challenge result

K=3 and K=5 tell coherent, hierarchically consistent stories — K=3's "premium" segment is K=5's clusters 2 and 3 merged. K=8 starts fragmenting natural groups, silhouette drops, and the marketing team would be designing 8 campaigns for segments that behave near-identically.

The takeaway is not that K=5 is "correct" — it is that **K is a modeling choice, not a ground truth discovery**. A DS practitioner's job is to match the cluster granularity to the business question. For broad campaign design, K=3. K=5 is a strong candidate for retention-focused segmentation because it isolates a high-value high-spend/low-income risk group, but final choice depends on business constraints.

---

## What surprised me

That elbow and silhouette agreed at K=5 on this dataset — they often disagree. In practice, disagreement is the more common and more interesting case (which I documented with a print note). Also: how much business meaning fell out of just 3 numeric features with no model. The "low-income, high-spend" segment practically screams for a retention campaign — and I found it without any supervised signal.

---

## What I don't fully understand yet

- t-SNE hyperparameter sensitivity: how much does perplexity actually change the cluster layout? I used 30 (default) but want to experiment with perplexity=5 vs 50 on a future dataset.
- DBSCAN: skipped it in the standard task (Mall Customers has spherical clusters, so K-Means is appropriate here). Will explore on a non-spherical dataset.
- UMAP vs t-SNE trade-offs beyond "UMAP is faster" — specifically when local vs global structure preservation matters more.

---

## Engineering Lesson: Consistency > Complexity

Initially, I introduced run-based file isolation (RUN_ID), but it created unnecessary complexity and broke pipeline consistency.

I reverted to a simpler contract-based structure:

- fixed filenames for intermediate data
- clear separation of data / models / outputs
- no dynamic path generation

This improved:

- reproducibility
- debugging speed
- pipeline reliability

## GitHub commit status: ✅

`day-13: clustering subjectivity study + customer profiling`

## Tomorrow's priority

Day 14 — Feature Engineering on Kaggle Housing Prices. First read the out-of-box challenge before writing a single line (anti-feature ablation is the hard part).
