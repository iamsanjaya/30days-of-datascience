import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_mnist_csv(csv_path, test_size, random_seed):
    """
    Loads raw MNIST CSV without ANY preprocessing.

    Responsibility:
    - read CSV
    - split dataset
    - return raw arrays only
    """

    df = pd.read_csv(csv_path)

    y = df["label"].to_numpy()
    X = df.drop(columns=["label"]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def one_hot_encode(y, num_classes):
    """
    Kept ONLY for NumPy scratch experiments if needed.
    Not used in Keras pipeline.
    """

    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded
