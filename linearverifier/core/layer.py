"""
This module defines the behavior of a neural network linear layer

"""
import numpy as np
from mpmath import mp


class Layer:
    pass


class LinearLayer(Layer):
    def __init__(self, weight: mp.matrix, bias: mp.matrix):
        super().__init__()

        self.weight = weight
        self.bias = bias

    def predict(self, x: mp.matrix) -> int:
        """Procedure to execute the matrix multiplication"""

        return np.argmax(self.weight * x + self.bias)
