"""Day 20 — CNN architecture builders.

build_overfit_model: a deliberately oversized CNN with zero regularization.
build_regularized_cnn: the same capacity, with Dropout / L2 / BatchNorm /
augmentation each independently toggleable — this is what lets
03_regularization_comparison.py isolate the effect of each fix.
"""

import tensorflow as tf

import config


def _augmentation_block() -> tf.keras.Sequential:
    """Random flip/rotation/zoom/translation — active only during model.fit(training=True)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(config.AUGMENTATION_ROTATION),
            tf.keras.layers.RandomZoom(config.AUGMENTATION_ZOOM),
            tf.keras.layers.RandomTranslation(
                config.AUGMENTATION_TRANSLATION, config.AUGMENTATION_TRANSLATION
            ),
        ],
        name="augmentation",
    )


def build_overfit_model(
    input_shape: tuple = config.INPUT_SHAPE, num_classes: int = config.NUM_CLASSES
) -> tf.keras.Model:
    """Deliberately oversized CNN, zero regularization. Built to overfit a tiny dataset."""
    return tf.keras.models.Sequential(
        name="overfit_baseline",
        layers=[
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
    )


def build_regularized_cnn(
    input_shape: tuple = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    dropout: float = 0.0,
    l2: float = 0.0,
    batchnorm: bool = False,
    augment: bool = False,
) -> tf.keras.Model:
    """Same capacity as build_overfit_model — each fix is an independent on/off switch."""
    reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None
    model_layers = [tf.keras.layers.Input(shape=input_shape)]

    if augment:
        model_layers.append(_augmentation_block())

    conv_filters = (64, 128, 256, 256)
    for i, filters in enumerate(conv_filters):
        model_layers.append(
            tf.keras.layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg)
        )
        if batchnorm:
            model_layers.append(tf.keras.layers.BatchNormalization())
        model_layers.append(tf.keras.layers.Activation("relu"))
        if i % 2 == 1:
            model_layers.append(tf.keras.layers.MaxPooling2D())

    model_layers.append(tf.keras.layers.Flatten())
    for _ in range(2):
        model_layers.append(tf.keras.layers.Dense(1024, kernel_regularizer=reg))
        if batchnorm:
            model_layers.append(tf.keras.layers.BatchNormalization())
        model_layers.append(tf.keras.layers.Activation("relu"))
        if dropout > 0:
            model_layers.append(tf.keras.layers.Dropout(dropout))

    model_layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return tf.keras.models.Sequential(model_layers, name="regularized_cnn")
