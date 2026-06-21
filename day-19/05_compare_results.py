"""
Day 19 — Step 5: Visualizations.
Generates (as PNGs, saved under outputs/):
  1. 3D scatter: depth x width x val_accuracy, faceted by dropout
  2. Heatmap grid: depth vs width, one panel per dropout rate
  3. Bar chart: full vs lottery-ticket vs random-pruned (accuracy + training time)

Run: python 05_compare_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import cast
import config
import numpy as np


def plot_3d_scatter(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors: dict[float, str] = {
        0.0: "tab:blue",
        0.2: "tab:orange",
        0.5: "tab:green",
    }

    for dropout, group in df.groupby("dropout"):
        dropout_value = cast(float, dropout)

        x = group["depth"].to_numpy(dtype=float)
        y = group["width"].to_numpy(dtype=float)
        z = group["val_accuracy"].to_numpy(dtype=float)

        ax.scatter(
            x.tolist(),
            y.tolist(),
            z.tolist(),
            label=f"dropout={dropout_value}",
            color=colors.get(dropout_value, "gray"),
            s=60,
        )

    ax.set_xlabel("Depth (hidden layers)")
    ax.set_ylabel("Width (neurons/layer)")
    ax.set_zlabel("Validation Accuracy")
    ax.set_title("Architecture Search: Depth x Width x Dropout -> Validation Accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(config.ARCH_SEARCH_DIR / "3d_architecture_grid.png", dpi=300)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame) -> None:
    dropouts = sorted(df["dropout"].unique())

    fig, axes = plt.subplots(
        1,
        len(dropouts),
        figsize=(6 * len(dropouts), 5),
        sharey=True,
    )

    if len(dropouts) == 1:
        axes = [axes]

    for ax, dropout in zip(axes, dropouts):
        subset = df[df["dropout"] == dropout]

        pivot = subset.pivot(
            index="depth",
            columns="width",
            values="val_accuracy",
        )

        im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")
        ax.set_title(f"dropout={float(dropout)}")

        fig.colorbar(im, ax=ax, label="Val Accuracy")

    fig.suptitle("Validation Accuracy Heatmap: Depth vs Width per Dropout Rate")
    fig.tight_layout()

    fig.savefig(config.ARCH_SEARCH_DIR / "depth_width_heatmaps.png", dpi=300)
    plt.close(fig)


def plot_pruning_comparison() -> None:
    df = pd.read_csv(config.PRUNING_COMPARISON_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = df["label"].tolist()
    acc = df["test_accuracy"].astype(float).tolist()

    axes[0].bar(labels, acc)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Full vs Lottery Ticket vs Random-Pruned")
    axes[0].set_ylim(0, 1)

    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plot_df = df.dropna(subset=["training_time_sec"])

    axes[1].bar(
        plot_df["label"].tolist(),
        plot_df["training_time_sec"].astype(float).tolist(),
    )
    axes[1].set_ylabel("Training Time (sec)")
    axes[1].set_title("Training Time (full model excluded)")

    fig.tight_layout()
    fig.savefig(config.PRUNING_OUTPUT_DIR / "pruning_comparison.png", dpi=300)
    plt.close(fig)


def main() -> None:
    grid_df = pd.read_csv(config.GRID_RESULTS_CSV)

    plot_3d_scatter(grid_df)
    plot_heatmaps(grid_df)

    if config.PRUNING_COMPARISON_CSV.exists():
        plot_pruning_comparison()
    else:
        print("Run pruning script first to generate comparison CSV.")

    print(f"Plots saved to {config.ARCH_SEARCH_DIR} and {config.PRUNING_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
