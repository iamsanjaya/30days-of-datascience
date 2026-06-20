import numpy as np


def init_params(input_size, hidden_size, output_size, seed):
    rng = np.random.default_rng(seed)

    W1 = rng.normal(0, np.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(
        np.float32
    )
    b1 = np.zeros((1, hidden_size), dtype=np.float32)

    W2 = rng.normal(0, np.sqrt(1.0 / hidden_size), (hidden_size, output_size)).astype(
        np.float32
    )
    b2 = np.zeros((1, output_size), dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(np.float32)


def softmax(z):
    z = z.astype(np.float32)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted, dtype=np.float32)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_pass(X, params):
    X = X.astype(np.float32)

    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def cross_entropy_loss(y_true_onehot, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    n = y_true_onehot.shape[0]
    return -np.sum(y_true_onehot * np.log(y_pred)) / n


def backward_pass(X, y_true_onehot, params, cache):
    n = X.shape[0]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = (A2 - y_true_onehot) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * relu_derivative(cache["Z1"])

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_params(params, grads, learning_rate):
    for k in params:
        params[k] -= learning_rate * grads["d" + k]
    return params


def predict(X, params):
    X = X.astype(np.float32)
    A2, _ = forward_pass(X, params)
    return np.argmax(A2, axis=1)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
