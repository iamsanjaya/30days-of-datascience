"""
utils/architecture.py — Frozen ResNet50 classifier head.

A single, fixed architecture is used across every subset-fraction experiment
in 02_train_subset_models.py and 03_augmentation_experiment.py so that the
*only* variable being studied is training-set size (and, separately,
augmentation) — not architecture differences.

Note: tf.keras.optimizers.legacy.Adam is required (not the standard Adam)
to avoid Metal graph remapper crashes on Apple Silicon when BatchNorm layers
are present — ResNet50's frozen backbone contains BatchNorm, so this applies
even though the backbone itself isn't being trained.

tensorflow-metal note: augmentation layers (RandomFlip/RandomRotation/
RandomZoom) must NOT live inside a tf.data.Dataset.map() call on Apple
Silicon — their internal random-seed state breaks when the dataset graph is
re-traced at each new epoch, causing a segfault partway through epoch 2.
The fix is to make augmentation part of the MODEL itself: Keras
automatically gates these layers to only run when training=True (i.e.
during model.fit()), so this is functionally identical to pipeline-level
augmentation without the Metal crash.
"""

import tensorflow as tf

import config


def build_frozen_resnet_classifier(
    img_size: tuple[int, int] = config.IMG_SIZE,
    learning_rate: float = config.FROZEN_BACKBONE_LR,
    augment: bool = False,
) -> tf.keras.Model:
    """Frozen ResNet50 backbone + trainable classification head (binary).

    augment=True prepends augmentation layers (active only during training)
    directly into the model graph — see the tensorflow-metal note above.
    """
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=img_size + (3,),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = inputs

    if augment:
        x = tf.keras.layers.RandomFlip("horizontal")(x)
        x = tf.keras.layers.RandomRotation(config.AUG_ROTATION)(x)
        x = tf.keras.layers.RandomZoom(config.AUG_ZOOM)(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model_name = (
        "frozen_resnet50_classifier_augmented"
        if augment
        else "frozen_resnet50_classifier"
    )
    model = tf.keras.Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
