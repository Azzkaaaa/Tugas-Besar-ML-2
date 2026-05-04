import numpy as np


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def apply_activation(x, activation=None):
    if activation is None:
        return x

    if activation == "relu":
        return relu(x)

    if activation == "softmax":
        return softmax(x)

    if activation == "sigmoid":
        return sigmoid(x)

    if activation == "tanh":
        return tanh(x)

    raise ValueError(f"Activation tidak dikenal: {activation}")