"""
This module defines the behavior of a neural network linear layer

"""

from linearverifier.core import ops


class Layer:
    pass


class LinearLayer(Layer):
    def __init__(self, weight: list[list[float]], bias: list[float]):
        super().__init__()

        self.weight = weight
        self.bias = bias

    def predict(self, x: list[float]) -> int:
        """Procedure to execute the matrix multiplication"""

        matmul = ops.matmul_right(self.weight, x)
        forward = [matmul[i] + self.bias[i] for i in range(len(self.bias))]

        return forward.index(max(forward))
