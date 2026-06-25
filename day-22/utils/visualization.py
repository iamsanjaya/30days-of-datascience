"""
utils/visualization.py — Plotting helpers.

Convention: never use plt.show() (silently fails to save on macOS).
Always plt.savefig() then plt.close().
"""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_data_efficiency_curve(results: list[dict], save_path: Path) -> None:
    """results: list of {"fraction": float, "train_size": int, "test_accuracy": float}."""
    results_sorted = sorted(results, key=lambda r: r["fraction"])
    fractions = [r["fraction"] * 100 for r in results_sorted]
    accuracies = [r["test_accuracy"] for r in results_sorted]
    train_sizes = [r["train_size"] for r in results_sorted]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(fractions, accuracies, marker="o", color="steelblue", linewidth=2)
    for x, y, n in zip(fractions, accuracies, train_sizes):
        ax.annotate(
            f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8
        )

    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "Accuracy Degrades as Training Data Shrinks (Cats vs Dogs, Frozen ResNet50)"
    )
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_augmentation_comparison(comparisons: list[dict], save_path: Path) -> None:
    """comparisons: list of {"fraction": float, "no_aug_accuracy": float, "aug_accuracy": float}."""
    labels = [f"{c['fraction'] * 100:.0f}%" for c in comparisons]
    no_aug = [c["no_aug_accuracy"] for c in comparisons]
    aug = [c["aug_accuracy"] for c in comparisons]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [i - width / 2 for i in x],
        no_aug,
        width,
        label="No augmentation",
        color="indianred",
    )
    ax.bar(
        [i + width / 2 for i in x],
        aug,
        width,
        label="With augmentation",
        color="seagreen",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training data used")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Augmentation Effect on Small Training Subsets")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_niche_comparison(results: dict[str, dict], save_path: Path) -> None:
    """results: {"baseline_cnn": {"accuracy": ...}, "cnn_augmented": {...}, ...}."""
    order = [
        "baseline_cnn",
        "cnn_augmented",
        "transfer_learning",
        "transfer_learning_tta",
        "pseudo_labeling",
    ]
    labels, accuracies = [], []
    for key in order:
        if key in results:
            labels.append(key.replace("_", "\n"))
            accuracies.append(results[key]["accuracy"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, accuracies, color="darkorange")
    for bar, acc in zip(bars, accuracies):
        ax.annotate(
            f"{acc:.2%}",
            (bar.get_x() + bar.get_width() / 2, acc),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Test accuracy")
    ax.set_title("Data Efficiency Techniques on a 450-Image Nepali Digit Dataset")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis="y")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
