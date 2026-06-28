"""
utils/architectures.py
Model architecture for the Word2Vec + LSTM path.

Note: this architecture has no BatchNorm layer, so the standard
tf.keras.optimizers.Adam is safe here on tensorflow-metal. The
legacy.Adam workaround is only needed when BatchNorm is present
(see learning-journal notes from the CNN days).
"""

import numpy as np
import tensorflow as tf

import config


def build_lstm_model(embedding_matrix: np.ndarray) -> tf.keras.Model:
    vocab_size, embedding_dim = embedding_matrix.shape

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=config.LSTM_MAX_LEN,
                trainable=True,  # fine-tune the Word2Vec vectors on the sentiment task
                mask_zero=True,
            ),
            tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=False),
            tf.keras.layers.Dropout(config.LSTM_DROPOUT),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(config.LSTM_DROPOUT),
            tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax"),
        ],
        name="word2vec_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
