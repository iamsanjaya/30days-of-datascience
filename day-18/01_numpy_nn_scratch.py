import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import config as _config
from utils.data_loader import load_mnist_csv, one_hot_encode
from utils.preprocess import prepare_data
from utils.nn_utils import (
    init_params,
    forward_pass,
    backward_pass,
    update_params,
    cross_entropy_loss,
    predict,
    accuracy,
)

sys.path.append(str(Path(__file__).resolve().parent))
config = cast(Any, _config)


def get_batches(X, y, batch_size, rng):
    n = X.shape[0]
    indices = rng.permutation(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train(X_train, y_train, X_test, y_test, params, epochs, batch_size, lr, seed):
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(1, epochs + 1):
        epoch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size, rng):
            y_pred, cache = forward_pass(X_batch, params)
            loss = cross_entropy_loss(y_batch, y_pred)

            grads = backward_pass(X_batch, y_batch, params, cache)
            params = update_params(params, grads, lr)

            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))

        test_pred = predict(X_test, params)
        test_acc = accuracy(y_test, test_pred)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_accuracy": test_acc,
            }
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3} | loss: {train_loss:.4f} | acc: {test_acc:.4f}")

    return params, history


def main():
    print("Loading MNIST data...")

    X_train, X_test, y_train, y_test = load_mnist_csv(
        config.TRAIN_CSV,
        config.TEST_SIZE,
        config.RANDOM_SEED,
    )

    # CRITICAL: enforce shared preprocessing contract
    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)

    # NumPy model requires one-hot labels
    y_train_onehot = one_hot_encode(y_train, config.NUM_CLASSES)

    params = init_params(
        config.INPUT_SIZE,
        config.HIDDEN_SIZE,
        config.OUTPUT_SIZE,
        config.RANDOM_SEED,
    )

    trained_params, history = train(
        X_train,
        y_train_onehot,
        X_test,
        y_test,
        params,
        epochs=config.NUMPY_EPOCHS,
        batch_size=config.NUMPY_BATCH_SIZE,
        lr=config.NUMPY_LEARNING_RATE,
        seed=config.RANDOM_SEED,
    )

    final_acc = history[-1]["test_accuracy"]
    print(f"\nFinal NumPy model accuracy: {final_acc:.4f}")

    pd.DataFrame(history).to_csv(
        config.METRICS_DIR / "numpy_nn_history.csv",
        index=False,
    )

    np.savez(
        config.MODELS_DIR / "numpy_nn_params.npz",
        W1=trained_params["W1"],
        b1=trained_params["b1"],
        W2=trained_params["W2"],
        b2=trained_params["b2"],
    )


if __name__ == "__main__":
    main()
