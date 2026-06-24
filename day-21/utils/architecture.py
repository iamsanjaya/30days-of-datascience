"""
Model architecture: a configurable backbone (ResNet50 or Xception,
ImageNet weights) + a small classification head. Two phases share this
same builder:
  Phase 1 — frozen base, train head only.
  Phase 2 — unfreeze the last N layers of the base, fine-tune end to end.

Which backbone gets built is driven entirely by config.BACKBONE — set
it once in config.py and everything else (image size, preprocessing,
Grad-CAM layer name) follows automatically.
"""

from __future__ import annotations

from typing import cast
import tensorflow as tf
from tensorflow.keras import (  # pyright: ignore[reportMissingModuleSource]
    layers,  # pyright: ignore[reportMissingModuleSource]
    models,  # pyright: ignore[reportMissingModuleSource]
)  # pyright: ignore[reportMissingModuleSource]

import config


def _build_base_model() -> tf.keras.Model:
    input_shape = config.IMG_SIZE + (config.CHANNELS,)
    if config.BACKBONE == "resnet50":
        return tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif config.BACKBONE == "xception":
        return tf.keras.applications.Xception(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown backbone in config: {config.BACKBONE}")


def build_model(trainable_base: bool = False) -> tf.keras.Model:
    """
    Build backbone + classification head.
    trainable_base=False -> Phase 1 (frozen feature extractor).
    trainable_base=True  -> base is set trainable, but caller should
                            still selectively freeze early layers via
                            unfreeze_top_layers() before compiling.
    """
    base_model = _build_base_model()
    base_model.trainable = trainable_base

    inputs = layers.Input(shape=config.IMG_SIZE + (config.CHANNELS,))
    x = base_model(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    outputs = layers.Dense(
        config.NUM_CLASSES, activation="sigmoid", name="predictions"
    )(x)

    model = models.Model(inputs, outputs, name=f"{config.BACKBONE}_transfer")
    return model


def get_base_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Retrieve the nested backbone layer by name. Using config.BACKBONE
    (Keras names the nested model "resnet50" or "xception" respectively)
    rather than a hardcoded string keeps this working for either backbone.
    """
    return cast(tf.keras.Model, model.get_layer(config.BACKBONE))


def unfreeze_top_layers(
    model: tf.keras.Model, n_layers: int = config.UNFREEZE_LAYERS
) -> tf.keras.Model:
    """
    Unfreeze the last n_layers of the backbone in-place, keeping all
    earlier layers frozen. BatchNorm layers are deliberately left
    frozen (kept in inference mode) even within the unfrozen block,
    which is standard practice for fine-tuning — retraining BN
    statistics on a small batch size destabilizes training.
    """
    base_model = get_base_model(model)
    base_model.trainable = True

    freeze_until = len(base_model.layers) - n_layers
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model
