from __future__ import annotations

import sys
from pathlib import Path

import config
from utils.data_loader import load_mnist_csv
from utils.preprocess import prepare_data

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_model(seed: int) -> keras.Model:
    keras.utils.set_random_seed(seed)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(config.INPUT_SIZE,)),
            keras.layers.Dense(
                config.HIDDEN_SIZE, activation="relu", name="hidden_dense"
            ),
            keras.layers.Dense(
                config.OUTPUT_SIZE, activation="softmax", name="output_dense"
            ),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=config.KERAS_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_random_label_model(X_train, X_test, y_train, y_test, seed: int):
    rng = np.random.default_rng(seed)
    y_train_shuffled = rng.permutation(y_train)

    model = build_model(seed)

    model.fit(
        X_train,
        y_train_shuffled,
        epochs=config.RANDOM_LABEL_EPOCHS,
        batch_size=config.KERAS_BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=2,
    )

    return model


def plot_first_layer_weights(
    model,
    title: str,
    save_path: Path,
    num_neurons: int = 25,
):
    # Get all Dense layers (robust across different model builds)
    dense_layers = [
        layer for layer in model.layers if isinstance(layer, keras.layers.Dense)
    ]

    if not dense_layers:
        raise ValueError("No Dense layer found in model")

    hidden_layer = dense_layers[0]
    weights = hidden_layer.get_weights()[0]

    grid_size = int(np.sqrt(num_neurons))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i >= weights.shape[1]:
            break

        neuron_weights = weights[:, i].reshape(28, 28)
        ax.imshow(neuron_weights, cmap="gray")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    print("Loading MNIST data...")

    X_train, X_test, y_train, y_test = load_mnist_csv(
        config.TRAIN_CSV,
        config.TEST_SIZE,
        config.RANDOM_SEED,
    )

    X_train, y_train = prepare_data(X_train, y_train)
    X_test, y_test = prepare_data(X_test, y_test)

    clean_model_path = config.MODELS_DIR / "keras_nn_replica.keras"

    if not clean_model_path.exists():
        raise FileNotFoundError("Run 02_keras_nn_replica.py first.")

    clean_model = keras.models.load_model(clean_model_path)

    print("Training random-label model...")

    random_label_model = train_random_label_model(
        X_train, X_test, y_train, y_test, config.RANDOM_SEED
    )

    random_label_model.save(config.MODELS_DIR / "keras_nn_random_label.keras")

    plot_first_layer_weights(
        clean_model,
        "Weights (Real Labels)",
        config.PLOTS_DIR / "weights_real_labels.png",
    )

    plot_first_layer_weights(
        random_label_model,
        "Weights (Shuffled Labels)",
        config.PLOTS_DIR / "weights_shuffled_labels.png",
    )


if __name__ == "__main__":
    main()
