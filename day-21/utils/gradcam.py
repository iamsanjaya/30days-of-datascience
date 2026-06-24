"""
Grad-CAM (Selvaraju et al., 2017): gradient-weighted class activation
maps. Shows which spatial regions of an image most influenced the
model's prediction, by backpropagating the gradient of the predicted
class score into the last convolutional feature map.

This is the core tool for the Day 21 out-of-box challenge: "What does
ResNet actually see?" — for correctly classified vs misclassified
images, is the model looking at the right region and still getting it
wrong, or looking at the wrong region entirely?
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib as mpl

import config


def make_gradcam_heatmap(
    img_array: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: str = config.LAST_CONV_LAYER_NAME,
) -> np.ndarray:
    """
    img_array: a single preprocessed image, shape (1, H, W, 3).
    Returns a 2D heatmap (values in [0, 1]) at the resolution of the
    last conv layer's feature map.
    """
    base_model = model.get_layer(config.BACKBONE)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Sub-model mapping inputs -> last conv activations.
    conv_model = tf.keras.Model(base_model.input, last_conv_layer.output)

    # Sub-model mapping conv activations -> final prediction, by
    # replaying the head layers that sit after the base model.
    head_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = head_input
    head_started = False
    for layer in model.layers:
        if layer.name == config.BACKBONE:
            head_started = True
            continue
        if head_started:
            x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        predictions = head_model(conv_output)
        class_score = predictions[:, 0]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)  # ReLU
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = config.GRADCAM_ALPHA,
) -> np.ndarray:
    """
    original_image: uint8 RGB array, shape (H, W, 3), in [0, 255].
    heatmap: 2D array in [0, 1], any resolution (will be resized).
    Returns an RGB uint8 overlay image, same shape as original_image.
    """

    h, w = original_image.shape[:2]
    heatmap_resized = (
        tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()
    )

    colormap = mpl.colormaps["jet"]
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]  # drop alpha channel
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    overlay = (alpha * colored_heatmap + (1 - alpha) * original_image).astype(np.uint8)
    return overlay
