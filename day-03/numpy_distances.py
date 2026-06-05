# DAY 3 - STandard Task: NumPy Distance Matrices from Scratch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)


# 1. Euclidean Distance
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    √(Σ (a₁ - b₁ )² )
    Measure straight - line distance in n- dimensional space.
    Sensitive to magnitude - large values dominate.
    """
    return np.sqrt(np.sum((a - b) ** 2))


# 2. Cosine Similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    (a . b)/ (‖a‖ . ‖b‖)
    Measure the angle between two vectors, ignoring magnitude.
    Range: [-1,1]. 1 = identical directions, 0 = orthogonal, -1 = opposite.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# 3. Row-wise Normalization


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Scales each row so it has unit L2 norm (nomr = 1)
    Equivalent to projecting each row onto the unit hypersphere.
    Used before cosine similarity comparisons.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # keepdims=True keeps shape (n,1) so broadcasting divides row-by-row
    return matrix / norms


# 4. Correlation Matrix
def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Pearson correlation between every pair of COLUMNS.
    Forumula: Cov(X,Y) / (σ_X . σ_Y)

    Steps (all vectorized):
    1. Center each column (subtract mean)
    2. Compute covariance matrix via dot product
    3. Divide by outer product of standard deviations
    """
    # Step 1: center columns
    centered = matrix - matrix.mean(axis=0)

    # Step 2: covariance matrix (using n-1 for unbiased estimate)
    n = matrix.shape[0]
    cov = (centered.T @ centered) / (n - 1)

    # Step 3: standard deviations of each column
    stds = np.sqrt(np.diag(cov))

    # Step 4: divide by outer product of stds ⟶  correlation

    corr = cov / np.outer(stds, stds)

    # CLip to [-1, 1] to fix floating point drift
    return np.clip(corr, -1.0, 1.0)


# Demo & Verification
if __name__ == "__main__":

    # Test vectors
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print("=" * 50)
    print("Distanace metrics demo")
    print("=" * 50)
    print(f"Vector a: {a}")
    print(f"Vector b: {b}\n")

    ed = euclidean_distance(a, b)
    cs = cosine_similarity(a, b)
    print(f"Euclidean Distance: {ed: .4f}")
    print(f"Cosine Similarity: {cs: .4f}")

    # Verify against numpy/scipy (sanity check)
    ed_np = np.linalg.norm(a - b)
    print(
        f"\nNumPy norm verify: {ed_np: .4f} √ "
        if np.isclose(ed, ed_np)
        else "x Mismatch"
    )

    # Test row normalization
    print("\n" + "=" * 50)
    print("Row-wise Normalization")
    print("=" * 50)
    M = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
    M_norm = normalize_rows(M)
    print(f"Original: \n{M}")
    print(f"Normalized: \n {M_norm.round(4)}")
    row_norms = np.linalg.norm(M_norm, axis=1)
    print(f"Row norms after: {row_norms.round(6)}")

    # Test correlation matrix
    print("\n" + "=" * 50)
    print("Correlation Matrix")
    print("=" * 50)
    data = np.random.randn(100, 4)

    # Make columns 0 & 1 correlated on purpose
    data[:, 1] = data[:, 0] * 2 * np.random.randn(100) * 0.5
    my_corr = correlation_matrix(data)
    np_corr = np.corrcoef(data.T)

    print("Manual correlation matrix:")
    print(my_corr.round(4))
    print("\nNumPy corrcoef verify:")
    print(np_corr.round(4))
    print(f"\nMax absolute difference: {np.abs(my_corr - np_corr).max(): .2e}")

    #   Visualization

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Day 3 - Numpy Distance Metrics from Scratch", fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # Plot 1: Euclidean vs Cosine for random vector pairs
    ax1 = fig.add_subplot(gs[0])
    n_pairs = 50
    v1 = np.random.randn(n_pairs, 5)
    v2 = np.random.randn(n_pairs, 5)
    euc_distances = np.array([euclidean_distance(v1[i], v2[i]) for i in range(n_pairs)])
    cos_sims = np.array([cosine_similarity(v1[i], v2[i]) for i in range(n_pairs)])
    ax1.scatter(
        euc_distances, cos_sims, alpha=0.6, color="steelblue", edgecolors="white", s=50
    )
    ax1.set_xlabel("Euclidean Distance")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Euclidean vs Cosine\n(50 random vector pairs)")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Plot 2: Row norms before/after normalization
    ax2 = fig.add_subplot(gs[1])
    rand_mat = np.random.randn(30, 5) * np.random.uniform(1, 20, (30, 1))
    norms_before = np.linalg.norm(rand_mat, axis=1)
    norms_after = np.linalg.norm(normalize_rows(rand_mat), axis=1)
    ax2.scatter(
        range(30), norms_before, label="Before", alpha=0.7, color="tomato", s=40
    )
    ax2.scatter(
        range(30),
        norms_after,
        label="After",
        alpha=0.7,
        color="seagreen",
        s=40,
        marker="^",
    )
    ax2.axhline(1, color="seagreen", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Row index")
    ax2.set_ylabel("L2 Norm")
    ax2.set_title("Row Norms Before/After\nNormalization")
    ax2.legend()

    # Plot 3: Correlation heatmap
    ax3 = fig.add_subplot(gs[2])
    labels = [f"F{i}" for i in range(4)]
    im = ax3.imshow(my_corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            ax3.text(
                j,
                i,
                f"{my_corr[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    plt.colorbar(im, ax=ax3, fraction=0.046)
    ax3.set_title("Manual Correlation Matrix\n(F0-F1 intentionally correlated)")

    plt.tight_layout()
    plt.savefig(
        "/Users/sanjayachaudhary/Desktop/projects/day-03/plots/distance_metrics.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("\nPlot saved: distance_metrics.png")
