"""
utils/training.py — Shared train/evaluate helpers.

Reporting trap reminder: with restore_best_weights=True, history.history[...][-1]
reflects the *last logged epoch*, not the restored best weights. Always call
model.evaluate() after fit() for the metric that actually gets reported/saved.
"""

from pathlib import Path
from typing import cast

import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    patience: int,
    model_name: str,
    models_dir: Path,
) -> tf.keras.callbacks.History:
    """Train with early stopping (restored best weights) + checkpointing."""
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{model_name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    return history


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict[str, float]:
    """Evaluate on held-out test data. Always call this explicitly — do not
    trust history[...][-1] when restore_best_weights=True was used."""
    results = model.evaluate(test_ds, verbose=0, return_dict=True)
    results = cast(dict[str, float], results)
    return {k: float(v) for k, v in results.items()}
