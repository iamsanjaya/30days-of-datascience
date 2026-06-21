"""
Day 19 — Step 2: Systematic architecture search.
Vary depth (1-5 layers), width (32-512 neurons), dropout (0, 0.2, 0.5).
Logs train accuracy, val accuracy, and training time per configuration.

Results are appended row-by-row to grid_results.csv so the script is
safe to resume if interrupted (already-tested configs are skipped).

Run: python 02_architecture_search.py
"""

import time
import config
import numpy as np
import pandas as pd
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
    )


def already_tested(results_path, depth, width, dropout) -> bool:
    if not results_path.exists():
        return False

    df = pd.read_csv(results_path)
    return (
        (df["depth"] == depth) & (df["width"] == width) & (df["dropout"] == dropout)
    ).any()


def append_result(results_path, row: dict):
    df_row = pd.DataFrame([row])
    if results_path.exists():
        df_row.to_csv(results_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(results_path, mode="w", header=True, index=False)


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    X_train, y_train, X_val, y_val = load_splits()
    input_dim = X_train.shape[1]

    total = len(config.DEPTHS) * len(config.WIDTHS) * len(config.DROPOUT_RATES)
    done = 0

    for depth in config.DEPTHS:
        for width in config.WIDTHS:
            for dropout in config.DROPOUT_RATES:
                done += 1

                if already_tested(config.GRID_RESULTS_CSV, depth, width, dropout):
                    print(
                        f"[{done}/{total}] depth={depth} width={width} dropout={dropout} -> skipped"
                    )
                    continue

                print(
                    f"[{done}/{total}] Training depth={depth} width={width} dropout={dropout}"
                )

                model = build_mlp(
                    input_dim,
                    depth,
                    width,
                    dropout,
                    learning_rate=config.SEARCH_LEARNING_RATE,
                )

                n_params = model.count_params()

                early_stop = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.SEARCH_PATIENCE,
                    restore_best_weights=True,
                )

                start = time.time()

                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=config.SEARCH_EPOCHS,
                    batch_size=config.SEARCH_BATCH_SIZE,
                    callbacks=[early_stop],
                    verbose=0,
                    shuffle=True,
                )

                elapsed = time.time() - start

                train_acc = float(history.history["accuracy"][-1])
                val_acc = float(max(history.history["val_accuracy"]))
                epochs_ran = len(history.history["loss"])

                row = {
                    "depth": depth,
                    "width": width,
                    "dropout": dropout,
                    "n_params": n_params,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "epochs_ran": epochs_ran,
                    "training_time_sec": elapsed,
                }

                append_result(config.GRID_RESULTS_CSV, row)

                print(
                    f"   -> val_accuracy={val_acc:.4f}, params={n_params}, time={elapsed:.1f}s"
                )

                keras.backend.clear_session()

    print(f"\nGrid search complete. Results in {config.GRID_RESULTS_CSV}")


if __name__ == "__main__":
    main()
