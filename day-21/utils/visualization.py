"""
Plotting utilities. Every function saves to disk via plt.savefig() and
calls plt.close() — no plt.show(), consistent with the headless
pipeline pattern used since Day 1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_combined_curves(
    frozen_history: dict,
    finetune_history: dict,
    save_path: Path,
) -> None:
    """
    Stitches frozen-phase and fine-tune-phase curves end to end on one
    timeline, so the accuracy jump at the unfreeze point is visible.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    frozen_epochs = len(frozen_history["loss"])
    finetune_epochs = len(finetune_history["loss"])
    x_frozen = list(range(frozen_epochs))
    x_finetune = list(range(frozen_epochs, frozen_epochs + finetune_epochs))

    axes[0].plot(x_frozen, frozen_history["val_loss"], label="frozen (val)")
    axes[0].plot(x_finetune, finetune_history["val_loss"], label="fine-tuned (val)")
    axes[0].axvline(frozen_epochs, color="gray", linestyle="--", label="unfreeze point")
    axes[0].set_title("Validation Loss — Frozen vs Fine-Tuned")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(x_frozen, frozen_history["val_accuracy"], label="frozen (val)")
    axes[1].plot(x_finetune, finetune_history["val_accuracy"], label="fine-tuned (val)")
    axes[1].axvline(frozen_epochs, color="gray", linestyle="--", label="unfreeze point")
    axes[1].set_title("Validation Accuracy — Frozen vs Fine-Tuned")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_metric_comparison(metrics: dict, save_path: Path) -> None:
    """
    metrics: {"frozen": {"accuracy": ..., "precision": ..., ...},
              "finetuned": {...}}
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (phase_name, phase_metrics) in enumerate(metrics.items()):
        values = [phase_metrics[m] for m in metric_names]
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=phase_name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("Frozen vs Fine-Tuned — Test Set Metrics")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_gradcam_grid(
    samples: list[dict],
    save_path: Path,
    title: str,
) -> None:
    """
    samples: list of dicts with keys:
        original (uint8 array), overlay (uint8 array),
        true_label (str), pred_label (str), confidence (float)
    Renders a 2-column grid: original | Grad-CAM overlay, one row per sample.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, sample in enumerate(samples):
        axes[i, 0].imshow(sample["original"])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sample["overlay"])
        axes[i, 1].set_title(
            f"Grad-CAM | true={sample['true_label']} "
            f"pred={sample['pred_label']} ({sample['confidence']:.2f})"
        )
        axes[i, 1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
