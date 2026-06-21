"""
Day 19 — Out-of-box challenge: The Lottery Ticket Hypothesis (Frankle & Carbin, 2019).

Procedure:
1. Load the best model's trained weights -> compute global magnitude-based
   pruning masks (prune the smallest PRUNING_FRACTION of weights per Dense kernel).
2. Build a fresh model with the SAME architecture, reset to the ORIGINAL
   random initial weights (saved in step 3), apply the mask.
3. Train this masked "winning ticket" candidate from scratch, re-applying
   the mask after every batch so pruned weights stay at zero throughout training.
4. Control: also build a RANDOM mask at the same sparsity (same initial
   weights, random pruning instead of magnitude pruning) to check whether
   magnitude pruning is actually doing something smarter than chance.
5. Compare full / lottery-ticket / random-pruned on accuracy and training time.

Run: python 04_lottery_ticket_pruning.py
"""

import json
import time
import config
import numpy as np
import pandas as pd
from typing import cast, Tuple, List, Dict, Any
from utils.architecture import build_mlp
import tensorflow as tf

keras = tf.keras


def load_splits():
    data = np.load(config.PROCESSED_DIR / "splits.npz")
    return (
        data["X_train"].astype(np.float32),
        data["y_train"].astype(np.float32),
        data["X_val"].astype(np.float32),
        data["y_val"].astype(np.float32),
        data["X_test"].astype(np.float32),
        data["y_test"].astype(np.float32),
    )


def load_best_config() -> dict:
    with open(config.BEST_MODEL_OUTPUT_DIR / "best_config.json") as f:
        return json.load(f)


def load_initial_weights() -> List[np.ndarray]:
    data = np.load(config.BEST_MODEL_INIT_WEIGHTS_PATH)
    return [data[f"w{i}"] for i in range(len(data.files))]


def compute_magnitude_masks(
    trained_weights: List[np.ndarray], fraction: float
) -> List[np.ndarray]:
    masks: List[np.ndarray] = []

    for w in trained_weights:
        if w.ndim == 2:
            flat = np.abs(w)
            threshold = np.quantile(flat, fraction)
            mask = (np.abs(w) > threshold).astype(np.float32)
        else:
            mask = np.ones_like(w, dtype=np.float32)
        masks.append(mask)

    return masks


def compute_random_masks(
    reference_masks: List[np.ndarray], seed: int
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    masks: List[np.ndarray] = []

    for ref in reference_masks:
        sparsity = 1.0 - float(ref.mean())
        random_vals = rng.random(ref.shape)
        threshold = np.quantile(random_vals, sparsity)
        mask = (random_vals > threshold).astype(np.float32)
        masks.append(mask)

    return masks


class ApplyMaskCallback(keras.callbacks.Callback):
    def __init__(self, masks: List[np.ndarray]):
        super().__init__()
        self.masks = masks

    def on_train_batch_end(self, batch, logs=None):
        weights = self.model.get_weights()
        self.model.set_weights([w * m for w, m in zip(weights, self.masks)])


def train_masked_model(
    input_dim: int,
    cfg: Dict[str, Any],
    initial_weights: List[np.ndarray],
    masks: List[np.ndarray],
    label: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    model = build_mlp(
        input_dim,
        cfg["depth"],
        cfg["width"],
        cfg["dropout"],
        learning_rate=config.PRUNING_LEARNING_RATE,
    )

    # rewind + apply mask
    model.set_weights([w * m for w, m in zip(initial_weights, masks)])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.PRUNING_PATIENCE,
        restore_best_weights=True,
    )

    mask_cb = ApplyMaskCallback(masks)

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.PRUNING_EPOCHS,
        batch_size=config.PRUNING_BATCH_SIZE,
        callbacks=[early_stop, mask_cb],
        verbose=2,
        shuffle=True,
    )

    elapsed = time.time() - start

    result = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc = cast(Tuple[float, float], result)

    epochs_ran = len(history.history["loss"])

    final_weights = model.get_weights()
    total = 0
    zeroed = 0

    for w in final_weights:
        total += w.size
        zeroed += np.count_nonzero(w == 0)

    achieved_sparsity = zeroed / total if total > 0 else 0.0

    print(
        f"[{label}] test_acc={float(test_acc):.4f}, "
        f"time={elapsed:.1f}s, epochs={epochs_ran}, "
        f"sparsity={achieved_sparsity:.3f}"
    )

    return model, {
        "label": label,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "training_time_sec": elapsed,
        "epochs_ran": epochs_ran,
        "achieved_sparsity": float(achieved_sparsity),
    }


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    input_dim = X_train.shape[1]

    cfg = load_best_config()

    # full model baseline
    full_model = keras.models.load_model(config.BEST_MODEL_PATH)
    full_loss, full_acc = full_model.evaluate(X_test, y_test, verbose=0)

    full_result = {
        "label": "full_model",
        "test_accuracy": float(full_acc),
        "test_loss": float(full_loss),
        "training_time_sec": None,
        "epochs_ran": None,
        "achieved_sparsity": 0.0,
    }

    print(f"[full_model] test_acc={float(full_acc):.4f}")

    trained_weights = full_model.get_weights()
    initial_weights = load_initial_weights()

    magnitude_masks = compute_magnitude_masks(trained_weights, config.PRUNING_FRACTION)

    _, lottery_result = train_masked_model(
        input_dim,
        cfg,
        initial_weights,
        magnitude_masks,
        "lottery_ticket",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    random_masks = compute_random_masks(magnitude_masks, config.RANDOM_SEED)

    _, random_result = train_masked_model(
        input_dim,
        cfg,
        initial_weights,
        random_masks,
        "random_pruned",
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    results_df = pd.DataFrame([full_result, lottery_result, random_result])
    results_df.to_csv(config.PRUNING_COMPARISON_CSV, index=False)

    print(f"\nSaved comparison to {config.PRUNING_COMPARISON_CSV}")
    print(results_df)


if __name__ == "__main__":
    main()
