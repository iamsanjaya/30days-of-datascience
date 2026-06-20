from __future__ import annotations

import numpy as np


def prepare_data(X, y):
    """
    Data contract layer (single source of truth)

    Guarantees:
    - numpy arrays only
    - float32 inputs
    - int64 labels
    - flattened inputs (N, 784)
    - normalized pixel values [0, 1]
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)

    X = X / 255.0

    return X, y


def validate_data(X, y, name: str = "DATA"):
    """
    Lightweight sanity checks for debugging training stalls
    """

    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name}: X is not numpy array")

    if not isinstance(y, np.ndarray):
        raise TypeError(f"{name}: y is not numpy array")

    if X.ndim != 2:
        raise ValueError(f"{name}: X must be 2D (N, features), got {X.shape}")

    if np.isnan(X).any():
        raise ValueError(f"{name}: X contains NaN values")

    if np.isinf(X).any():
        raise ValueError(f"{name}: X contains Inf values")
