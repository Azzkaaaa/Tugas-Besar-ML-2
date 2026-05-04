import numpy as np
from src.common.activations import apply_activation

class Dense:
    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, x):
        out = np.dot(x, self.weights) + self.bias
        return apply_activation(out, self.activation)