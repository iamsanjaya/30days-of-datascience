"""Day 20 — Plotting helpers for training curves and the LR range test."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_overfit_curves(
    history: Dict[str, List[float]], save_name: str = "overfit_baseline.png"
) -> Path:
    """Single-config loss + accuracy curves — the classic overfitting signature."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(history["loss"], label="train")
    ax_loss.plot(history["val_loss"], label="val")
    ax_loss.set_title("Loss — deliberately overfit baseline")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_acc.plot(history["accuracy"], label="train")
    ax_acc.plot(history["val_accuracy"], label="val")
    ax_acc.set_title("Accuracy — deliberately overfit baseline")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()

    fig.tight_layout()
    out_path = config.OUTPUTS_DIR / save_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_regularization_grid(
    histories: Dict[str, Dict[str, List[float]]],
    save_name: str = "regularization_comparison.png",
) -> Path:
    """Side-by-side loss + accuracy curves for every regularization config."""
    n = len(histories)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharey="row")

    for col, (config_name, history) in enumerate(histories.items()):
        ax_loss, ax_acc = axes[0, col], axes[1, col]

        ax_loss.plot(history["loss"], label="train")
        ax_loss.plot(history["val_loss"], label="val")
        ax_loss.set_title(config_name)
        ax_loss.set_xlabel("epoch")
        if col == 0:
            ax_loss.set_ylabel("loss")
        ax_loss.legend(fontsize=8)

        ax_acc.plot(history["accuracy"], label="train")
        ax_acc.plot(history["val_accuracy"], label="val")
        ax_acc.set_xlabel("epoch")
        if col == 0:
            ax_acc.set_ylabel("accuracy")

    fig.suptitle("Overfit CNN — train/val curves before and after each fix")
    fig.tight_layout()
    out_path = config.OUTPUTS_DIR / save_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def smooth_ema(
    values: List[float], beta: float = config.LR_SMOOTHING_BETA
) -> np.ndarray:
    """Bias-corrected exponential moving average — flattens per-batch loss noise."""
    smoothed = []
    avg = 0.0
    for i, v in enumerate(values):
        avg = beta * avg + (1 - beta) * v
        smoothed.append(avg / (1 - beta ** (i + 1)))
    return np.array(smoothed)


def find_lr_range_test_points(
    lrs: List[float], losses: List[float]
) -> Dict[str, float]:
    """Identify the three landmark LRs from a range test sweep.

    - lr_start_decreasing: where smoothed loss first drops meaningfully
    - lr_steepest_descent: LR at the steepest negative gradient — the best
      candidate learning rate to actually train with
    - lr_diverge: LR where smoothed loss exceeds LR_DIVERGE_FACTOR x running min
    """
    smoothed = smooth_ema(losses)
    gradients = np.gradient(smoothed)

    initial_loss = smoothed[0]
    start_idx = next((i for i, v in enumerate(smoothed) if v < initial_loss * 0.98), 0)

    steepest_idx = int(np.argmin(gradients))

    running_min = np.minimum.accumulate(smoothed)
    diverge_candidates = np.where(smoothed > config.LR_DIVERGE_FACTOR * running_min)[0]
    diverge_idx = (
        int(diverge_candidates[0]) if len(diverge_candidates) else len(smoothed) - 1
    )

    return {
        "lr_start_decreasing": lrs[start_idx],
        "lr_steepest_descent": lrs[steepest_idx],
        "lr_diverge": lrs[diverge_idx],
    }


def plot_lr_range_test(
    lrs: List[float],
    losses: List[float],
    landmarks: Dict[str, float],
    save_name: str = "lr_range_test.png",
) -> Path:
    smoothed = smooth_ema(losses)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses, alpha=0.3, label="raw loss")
    ax.plot(lrs, smoothed, label="smoothed loss", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("loss")
    ax.set_title("LR Range Test — one epoch, exponential LR ramp")

    colors = {
        "lr_start_decreasing": "tab:green",
        "lr_steepest_descent": "tab:orange",
        "lr_diverge": "tab:red",
    }
    for label, lr_value in landmarks.items():
        ax.axvline(
            lr_value,
            color=colors[label],
            linestyle="--",
            label=f"{label} = {lr_value:.2e}",
        )

    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = config.OUTPUTS_DIR / save_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
