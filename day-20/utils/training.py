"""Day 20 — Training helpers: compile/fit boilerplate, history I/O, callbacks."""

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import tensorflow as tf

import config


def compile_model(
    model: tf.keras.Model, learning_rate: float = 1e-3, clipnorm: "float | None" = None
) -> tf.keras.Model:
    """Compile with Adam. clipnorm is opt-in — leave it None for the LR range
    test (04), which needs to observe genuine divergence at high LR. Pass it
    explicitly for 02/03, where it's a numerical-stability fix, not a
    capacity-limiting regularizer.

    Uses tf.keras.optimizers.legacy.Adam, not the default tf.keras.optimizers.Adam.
    On Apple Silicon + tensorflow-metal, the default Adam's op pattern both
    runs slower (TF's own runtime warning) and, combined with BatchNorm's
    extra moving-mean/variance update ops, can crash the Metal graph
    remapper outright. The legacy optimizer avoids both.
    """
    optimizer_kwargs: Dict[str, Any] = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(**optimizer_kwargs),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History, name: str) -> Path:
    """Persist a Keras History as JSON so later scripts can reuse it without retraining."""
    clean = {key: [float(v) for v in values] for key, values in history.history.items()}
    out_path = config.HISTORIES_DIR / f"{name}.json"
    out_path.write_text(json.dumps(clean, indent=2))
    return out_path


def load_history(name: str) -> Dict[str, List[float]]:
    path = config.HISTORIES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved history for '{name}' at {path}. Run the script that produces it first."
        )
    return json.loads(path.read_text())


def early_stopping_callback() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )


class LRRangeTestCallback(tf.keras.callbacks.Callback):
    """LR Range Test (Leslie Smith, 2017): ramp the learning rate from start_lr
    to end_lr over one epoch and record the per-batch loss.

    The ramp is exponential (log-linear), not linear in raw LR units.
    start_lr -> end_lr here spans eight orders of magnitude (1e-7 to 10) — a
    literal linear ramp would spend nearly every batch crammed near end_lr and
    barely sample the low end. Exponential spacing covers the search space
    evenly, which is what the technique actually relies on.
    """

    def __init__(self, start_lr: float, end_lr: float, total_iterations: int):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_iterations = max(total_iterations, 1)
        self.lr_multiplier = (end_lr / start_lr) ** (1 / self.total_iterations)
        self.lrs: List[float] = []
        self.losses: List[float] = []
        self._iteration = 0

    @property
    def _keras_model(self) -> tf.keras.Model:
        """Typed accessor for self.model.

        Callback.model is typed Optional[Model] in the Keras stubs because a
        callback can technically exist before being attached — but Keras
        always sets it before on_train_begin fires, so by the time any
        method below runs it is guaranteed non-None.
        """
        assert (
            self.model is not None
        ), "Callback was not attached to a model before training began"
        return self.model

    @property
    def _optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Typed accessor for self.model.optimizer.

        Model.optimizer is typed Optional[Optimizer] because it's None until
        .compile() is called — every script attaches this callback to an
        already-compiled model (see training.compile_model), so it's
        guaranteed non-None by the time training starts.
        """
        optimizer = self._keras_model.optimizer
        assert (
            optimizer is not None
        ), "Model must be compiled before attaching LRRangeTestCallback"
        return optimizer

    def on_train_begin(self, logs=None):
        self._optimizer.learning_rate.assign(self.start_lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = cast(Dict[str, float], logs)  # Keras always passes a populated dict here

        optimizer = self._optimizer
        current_lr = float(optimizer.learning_rate.numpy())
        self.lrs.append(current_lr)
        self.losses.append(float(logs["loss"]))

        self._iteration += 1
        next_lr = self.start_lr * (self.lr_multiplier**self._iteration)
        optimizer.learning_rate.assign(min(next_lr, self.end_lr))

        # Stop early once loss has clearly diverged — no point sweeping further.
        if len(self.losses) > 10 and self.losses[-1] > config.LR_DIVERGE_FACTOR * min(
            self.losses
        ):
            self._keras_model.stop_training = True
