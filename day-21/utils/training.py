"""
Training utilities: compile, attach callbacks, train, and persist
history as JSON (so 04_compare_results.py and visualization.py can
reload curves without retraining).
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

import config


def get_callbacks(checkpoint_path: Path) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-8,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    # tensorflow-metal note: legacy.Adam avoids the graph remapper
    # crashes seen with BatchNorm-heavy models (ResNet50 qualifies) on
    # this M4 / tensorflow-metal setup — same fix applied in Day 19/20.
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    checkpoint_path: Path,
) -> dict:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(checkpoint_path),
    )
    return history.history


def save_history(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cast numpy floats to plain floats for JSON serialization.
    clean = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def load_history(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
