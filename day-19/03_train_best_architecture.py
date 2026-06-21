"""
Day 19 — Step 3: Select the best architecture from the grid search,
train it fully, and save BOTH the trained model and its random
initial weights (needed later for lottery ticket pruning — the
hypothesis requires rewinding to the *original* initialization,
not a fresh random one).

Run: python 03_train_best_architecture.py
"""

import json
import config
import numpy as np
import pandas as pd

from typing import cast, Tuple
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


def select_best_config() -> dict:
    df = pd.read_csv(config.GRID_RESULTS_CSV)

    best_row = df.sort_values(config.BEST_MODEL_SELECTION_METRIC, ascending=False).iloc[
        0
    ]

    return {
        "depth": int(best_row.depth),
        "width": int(best_row.width),
        "dropout": float(best_row.dropout),
    }


def save_initial_weights(model: keras.Model, path) -> None:
    weights = model.get_weights()
    np.savez_compressed(path, **{f"w{i}": w for i, w in enumerate(weights)})


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    input_dim = X_train.shape[1]

    best_cfg = select_best_config()
    print(f"Best architecture from search: {best_cfg}")

    config_path = config.BEST_MODEL_OUTPUT_DIR / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

    model = build_mlp(
        input_dim,
        best_cfg["depth"],
        best_cfg["width"],
        best_cfg["dropout"],
        learning_rate=config.FINAL_LEARNING_RATE,
    )

    # Save initial weights BEFORE training (lottery ticket baseline)
    save_initial_weights(model, config.BEST_MODEL_INIT_WEIGHTS_PATH)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.FINAL_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.FINAL_EPOCHS,
        batch_size=config.FINAL_BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
        shuffle=True,
    )

    result = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc = cast(Tuple[float, float], result)

    print(
        f"Final test accuracy: {float(test_acc):.4f}, test loss: {float(test_loss):.4f}"
    )

    model.save(config.BEST_MODEL_PATH)

    pd.DataFrame(history.history).to_csv(
        config.BEST_MODEL_OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    with open(config.BEST_MODEL_OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
            },
            f,
            indent=2,
        )

    print(f"Saved best model to {config.BEST_MODEL_PATH}")
    print(f"Saved initial weights to {config.BEST_MODEL_INIT_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
