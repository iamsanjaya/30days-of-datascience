"""
utils/training.py
Shared training/evaluation plumbing used by all three model scripts so that
02/03/04 stay thin pipeline scripts instead of duplicating boilerplate.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Any, Dict, cast

import config


@contextmanager
def timer():
    """Usage: with timer() as t: ...   then t['seconds'] after the block exits."""
    state = {}
    start = time.perf_counter()
    yield state
    state["seconds"] = time.perf_counter() - start


def train_keras_model(
    model: tf.keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    checkpoint_path: Path,
    epochs: int = config.LSTM_EPOCHS,
    batch_size: int = config.LSTM_BATCH_SIZE,
    patience: int = config.LSTM_PATIENCE,
):
    """Fit with early stopping + best-checkpoint restore.

    Important: when restore_best_weights=True, history.history[...][-1] reflects
    the LAST epoch trained, not the restored best-weight epoch. Always call
    model.evaluate() after fit() to get the true restored-weight performance —
    do not read it off the history object.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return history


def plot_keras_training_curves(history, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


def compute_and_save_metrics(
    model_name: str,
    y_true,
    y_pred,
    train_seconds: float,
    inference_seconds: float,
    model_size_mb: float,
) -> dict:
    """Compute accuracy/F1/report, save confusion matrix + metrics JSON, return the dict.

    This dict is the contract that 05_compare_models.py reads — every model
    script (02/03/04) must call this with the same keys.
    """
    report = cast(
        Dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            target_names=config.CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    )
    metrics = {
        "model_name": model_name,
        "accuracy": float(report["accuracy"]),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_1000": (inference_seconds / max(len(y_true), 1)) * 1000,
        "model_size_mb": model_size_mb,
        "classification_report": report,
    }

    metrics_path = config.OUTPUT_DIR / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(
        y_true,
        y_pred,
        config.OUTPUT_DIR / f"confusion_matrix_{model_name}.png",
        title=f"{model_name} — Confusion Matrix",
    )

    return metrics


def model_size_on_disk_mb(path: Path) -> float:
    """Sum file size(s) on disk in MB. Accepts a single file or a directory (e.g. HF checkpoints)."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)
