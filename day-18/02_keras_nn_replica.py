from __future__ import annotations

import sys
from pathlib import Path

import config
from utils.data_loader import load_mnist_csv

import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cfg(name: str):
    if not hasattr(config, name):
        raise RuntimeError(f"Missing config value: {name}")
    return getattr(config, name)


def build_model(input_size, hidden_size, output_size, seed):
    tf.random.set_seed(seed)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(output_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    print("Loading MNIST data...")

    X_train, X_test, y_train, y_test = load_mnist_csv(
        cfg("TRAIN_CSV"),
        cfg("TEST_SIZE"),
        cfg("RANDOM_SEED"),
    )

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    model = build_model(
        cfg("INPUT_SIZE"),
        cfg("HIDDEN_SIZE"),
        cfg("OUTPUT_SIZE"),
        cfg("RANDOM_SEED"),
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=cfg("KERAS_EPOCHS"),
        batch_size=cfg("KERAS_BATCH_SIZE"),
        validation_data=(X_test, y_test),
        shuffle=True,
        verbose=1,
    )

    eval_result = model.evaluate(X_test, y_test, verbose=0)

    if isinstance(eval_result, (list, tuple)):
        test_acc = eval_result[1]
    else:
        test_acc = eval_result
    print(f"\nFinal Keras replica test accuracy: {test_acc:.4f}")
    print(f"\nFinal Keras replica test accuracy: {test_acc:.4f}")

    metrics_dir = cfg("METRICS_DIR")
    models_dir = cfg("MODELS_DIR")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history.history).to_csv(
        metrics_dir / "keras_nn_history.csv",
        index=False,
    )

    model.save(models_dir / "keras_nn_replica.keras")

    print(f"Saved history -> {metrics_dir / 'keras_nn_history.csv'}")
    print(f"Saved model   -> {models_dir / 'keras_nn_replica.keras'}")


if __name__ == "__main__":
    main()
