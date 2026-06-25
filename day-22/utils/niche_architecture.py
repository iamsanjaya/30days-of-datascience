"""
utils/niche_architecture.py — Models for the 450-image Nepali digit challenge.

build_tiny_cnn          — small CNN trained from scratch (the "no tricks" baseline)
build_transfer_model    — frozen MobileNetV2 backbone + head (the transfer-learning trick)
unfreeze_for_fine_tuning — unfreezes the top of the backbone for a low-LR fine-tune pass

MobileNetV2 is used instead of ResNet50 here because it's lighter and was
designed to work well at smaller input resolutions — a better fit for
96x96 upscaled 32x32 source images than a deeper, higher-resolution-tuned
network like ResNet50.

tf.keras.optimizers.legacy.Adam is used throughout — both architectures
contain BatchNorm layers, which crash the Metal graph remapper with the
standard Adam optimizer on Apple Silicon.
"""

import tensorflow as tf

import config


def build_tiny_cnn(
    img_size: tuple[int, int] = config.NICHE_IMG_SIZE,
    num_classes: int = config.NICHE_NUM_CLASSES,
    learning_rate: float = config.NICHE_HEAD_LR,
) -> tf.keras.Model:
    """Small CNN trained from scratch — the baseline with no efficiency tricks."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=img_size + (3,)),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="tiny_cnn_baseline",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    img_size: tuple[int, int] = config.NICHE_IMG_SIZE,
    num_classes: int = config.NICHE_NUM_CLASSES,
    learning_rate: float = config.NICHE_HEAD_LR,
) -> tf.keras.Model:
    """Frozen MobileNetV2 backbone + trainable head."""
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=img_size + (3,),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    # MobileNetV2 expects [-1, 1] inputs; our niche_data loader scales to [0, 1].
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    num_layers_to_unfreeze: int = 20,
    learning_rate: float = config.NICHE_FINE_TUNE_LR,
) -> tf.keras.Model:
    """Unfreeze the top N layers of the MobileNetV2 backbone for a low-LR fine-tune pass."""
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise RuntimeError("Could not locate MobileNetV2 backbone layer to unfreeze.")

    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
