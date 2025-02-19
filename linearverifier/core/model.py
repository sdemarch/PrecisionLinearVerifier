"""
This module defines the behavior of a neural network model

"""

from linearverifier.core import ops
from linearverifier.core.layer import LinearLayer
from linearverifier.parser import onnx
from linearverifier.parser import vnnlib

DATA_PATH = 'Data/MNIST'


class ModelOptions:
    pass


class LinearModel:
    def __init__(self, onnx_path: str, w_path=f'{DATA_PATH}/mnist_weights.txt', b_path=f'{DATA_PATH}/mnist_bias.txt',
                 options: ModelOptions = None):
        self.onnx_path = onnx_path

        self.w_path = w_path
        self.b_path = b_path
        self.options = options

        self.layer = self.parse_layer()

    @staticmethod
    def check_robust(lbs: list[float], ubs: list[float], label: int) -> bool:
        """Procedure to check whether the robustness specification holds using numeric bounds"""

        # Create the matrices of the disjunctions
        out_props = ops.create_disjunction_matrix(len(lbs), label)
        bounds = {
            'lower': lbs,
            'upper': ubs
        }

        # For each disjunction in the output property, check none is satisfied by output_bounds.
        # If one disjunction is satisfied, then it represents a counter-example.
        for i in range(len(out_props)):
            result = ops.check_unsafe(bounds, out_props[i])

            if result:
                return False  # unsafe -> not robust

        return True

    @staticmethod
    def check_sym_robust(in_lbs: list[float], in_ubs: list[float], sym_bounds: tuple, label: int) -> bool:
        """Procedure to check whether the robustness specification holds using symbolic bounds"""

        # Propagate property as a layer
        out_props = ops.create_disjunction_matrix(len(sym_bounds[0]), label)

        # For each disjunction in the output property, check none is satisfied by output_bounds.
        # If one disjunction is satisfied, then it represents a counter-example.
        for i in range(len(out_props)):
            """Check intersection with symbolic bounds"""
            result = ops.check_unsafe_sym(sym_bounds, out_props[i], in_lbs, in_ubs)

            if result:
                return False

        return True

    def parse_layer(self) -> LinearLayer:
        """Procedure to read the first layer of a ONNX network"""
        nn = onnx.nn_from_weights(self.w_path, self.b_path)  # nn_from_onnx(self.onnx_path)
        return nn[0]

    def propagate(self, lbs: list[float], ubs: list[float]) -> tuple[list[float], list[float]]:
        """Procedure to compute the numeric interval bounds of a linear layer"""
        weights_plus = ops.get_positive(self.layer.weight)
        weights_minus = ops.get_negative(self.layer.weight)

        wpluslb = ops.matmul_right(weights_plus, lbs)
        wplusub = ops.matmul_right(weights_plus, ubs)
        wminlb = ops.matmul_right(weights_minus, lbs)
        wminub = ops.matmul_right(weights_minus, ubs)

        low = [wpluslb[i] + wminub[i] + self.layer.bias[i] for i in range(len(self.layer.bias))]
        upp = [wplusub[i] + wminlb[i] + self.layer.bias[i] for i in range(len(self.layer.bias))]

        return low, upp

    def verify(self, vnnlib_path: str) -> bool:
        # 1: Read VNNLIB bounds
        in_lbs, in_ubs, label = vnnlib.read_vnnlib(vnnlib_path)

        # 2: Propagate input through linear layer
        out_lbs, out_ubs = self.propagate(in_lbs, in_ubs)

        # 3: Check intersection
        numeric_check = LinearModel.check_robust(out_lbs, out_ubs, label)
        if numeric_check:
            return True
        else:
            return LinearModel.check_sym_robust(in_lbs, in_ubs, (self.layer.weight, self.layer.bias), label)
