# Day 03 — June 05, 2025

**Date:** 2025-06-06
**Phase:** 1 — Foundation
**Topic:** NumPy — Arrays, Broadcasting, Vectorization & Distance Metrics

---

## What I Built Today

**Standard Task — `numpy_distances.py`**
Implemented four core distance/similarity operations from scratch using only NumPy (zero loops):

- Euclidean distance between two vectors
- Cosine similarity between two vectors
- Row-wise L2 normalization of a matrix
- Pearson correlation matrix via centered dot product

Verified all implementations against NumPy's own functions — max absolute difference was `1.11e-16` (pure floating point noise). Saved a 3-panel visualization to `plots/distance_metrics.png`.

**Out-of-Box Challenge — `distance_perception.py`**
Built two concrete examples proving that Euclidean and Cosine metrics can give _opposite_ answers about closeness:

- **Example 1 (Euclidean-Close, Cosine-Far):** User A (Comedy lover) and User B (Action lover) are Euclidean-close (`9.899`) but Cosine-far (`0.246`) — opposite tastes. A recommender using Euclidean would serve the wrong movies.
- **Example 2 (Euclidean-Far, Cosine-Close):** A harsh critic (D) and a generous rater (E) are Euclidean-far (`13.416`) but Cosine-close (`0.972`) — identical preferences. Euclidean sees scale difference; Cosine sees agreement.

Wrote a 10-line ML explanation of when to choose each metric. Saved visualization to `plots/distance_perception.png`.

---

## The Out-of-Box Challenge Result

The hardest part was finding examples that were _mathematically clean_ — not just conceptually right but numerically convincing. The movie rating scenario worked perfectly because it maps directly to a real recommendation system problem.

The key insight that clicked: **Cosine similarity is scale-invariant by design.** Normalizing a vector before computing dot product removes magnitude entirely — you're left with pure direction. This is exactly why word embeddings use cosine: "happy" and "joyful" might have different L2 norms depending on training frequency, but their directions in embedding space should be nearly identical.

The 10-line explanation forced me to think about _when magnitude is information_ vs _when magnitude is noise_. That distinction is the real decision rule — not "use cosine for text" as a memorized rule.

---

## What Surprised Me

**1. Broadcasting saved the entire correlation matrix implementation.**
The line `cov / np.outer(stds, stds)` divides a `(4,4)` matrix by a `(4,4)` outer product in one operation — no loops, no indexing. I had originally written a nested for-loop version first, then replaced it. The vectorized version is not just faster — it's actually _easier to read_ once you understand what `np.outer` does.

**2. Cosine similarity can be negative.**
I knew the range was `[-1, 1]` but didn't intuitively feel what a negative cosine meant until the movie rating example. User A (Comedy=8, Action=1) and a hypothetical User Z (Comedy=1, Action=8 in a higher-dimensional space) would have cosine close to `-1` — literally pointing in opposite directions. The metric encodes _opposition_, not just similarity.

**3. The correlation matrix clipping.**
`np.clip(corr, -1.0, 1.0)` felt like a hack at first. But floating point arithmetic on the division step can produce values like `1.0000000000000002` due to rounding — which would make a correlation matrix technically invalid. Clipping is the correct engineering fix, not a workaround.

---

## What I Don't Fully Understand Yet

**1. When does cosine similarity break down in high dimensions?**
I know the "curse of dimensionality" exists and that distances behave strangely in very high-dimensional spaces. But I don't have a clear intuition yet for _at what dimensionality_ cosine similarity starts losing meaning, or what the alternative is (random projections? inner product spaces?). This needs more reading.

**2. The angle arc visualization in the out-of-box plot.**
The arc I drew between vectors A and B using parametric equations doesn't perfectly bisect the angle — the scaling by `user_A1[0] / np.linalg.norm(user_A1)` was an approximation that looked reasonable visually but isn't geometrically precise. I need to revisit how to properly draw angle arcs between two arbitrary 2D vectors.

**3. Unbiased vs biased covariance.**
I used `n - 1` (Bessel's correction) in the covariance computation because that's the standard for sample covariance — and it matched `np.corrcoef`. But I don't have a deep probabilistic intuition yet for _why_ `n - 1` makes the estimator unbiased. It's on the list.

---

## GitHub Commit Made: ✅

`Day 03 — NumPy Distance Metrics from Scratch + Distance -is-perception challenge`

---

## Tomorrow's Priority

**Day 4 — Gradient Descent from Scratch**

- Implement batch gradient descent for linear regression using only NumPy
- Train on synthetic dataset, plot loss curve + predicted vs actual
- Experiment with 3 learning rates — document divergence behavior
- Out-of-box: find the exact LR threshold where training breaks, explain it geometrically
