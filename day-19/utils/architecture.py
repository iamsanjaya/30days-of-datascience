"""
Shared model-building utility used by the architecture search,
final training, and pruning scripts. Not a pipeline stage itself,
hence no numeric prefix.
"""

import tensorflow as tf

keras = tf.keras


def build_mlp(
    input_dim: int,
    depth: int,
    width: int,
    dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    for _ in range(depth):
        x = keras.layers.Dense(width, activation="relu")(x)

        if dropout > 0.0:
            x = keras.layers.Dropout(float(dropout))(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
