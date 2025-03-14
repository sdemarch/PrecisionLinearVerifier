"""
This module defines the behavior of a neural network model

"""

from linearverifier.core import ops
from linearverifier.core.layer import LinearLayer
from linearverifier.parser import onnx
from linearverifier.parser import vnnlib
from mpmath import mp

MNIST_PATH = 'Data/MNIST'


class ModelOptions:
    pass


class LinearModel:
    def __init__(self, onnx_path: str, options: ModelOptions = None):

        self.onnx_path = onnx_path
        self.options = options

        self.layer = self.parse_layer()

    @staticmethod
    def check_robust(lbs: mp.matrix, ubs: mp.matrix, label: int) -> bool:
        """Procedure to check whether the robustness specification holds using numeric bounds"""

        # Create the matrices of the disjunctions
        out_props = ops.create_disjunction_matrix(lbs.rows, label)
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
    def check_sym_robust(in_lbs: mp.matrix, in_ubs: mp.matrix, sym_bounds: dict, label: int) -> tuple[bool, int | None]:
        """Procedure to check whether the robustness specification holds using symbolic bounds"""

        # Propagate property as a layer
        out_props = ops.create_disjunction_matrix(sym_bounds['matrix'].rows, label)

        # For each disjunction in the output property, check none is satisfied by output_bounds.
        # If one disjunction is satisfied, then it represents a counter-example.
        for i in range(len(out_props)):
            """Check intersection with symbolic bounds"""
            result = ops.check_unsafe_sym(sym_bounds, out_props[i], in_lbs, in_ubs)

            if result:
                if len(out_props[i]) == 1:
                    return False, None
                else:
                    return False, list(out_props[i]).index(-1)

        return True, None

    def parse_layer(self) -> LinearLayer:
        """Procedure to read the first layer of a ONNX network"""
        nn = onnx.nn_from_onnx(self.onnx_path)
        return nn[0]

    def propagate(self, lbs: mp.matrix, ubs: mp.matrix) -> tuple[mp.matrix, mp.matrix]:
        """Procedure to compute the numeric interval bounds of a linear layer"""
        weights_plus = ops.get_positive(self.layer.weight)
        weights_minus = ops.get_negative(self.layer.weight)

        low = weights_plus * lbs + weights_minus * ubs + self.layer.bias
        upp = weights_plus * ubs + weights_minus * lbs + self.layer.bias

        return low, upp

    def verify(self, vnnlib_path: str) -> tuple[bool, int | None]:
        # 1: Read VNNLIB bounds
        in_lbs, in_ubs, label = vnnlib.read_vnnlib(vnnlib_path)

        # 2: Propagate input through linear layer
        out_lbs, out_ubs = self.propagate(in_lbs, in_ubs)

        # 3: Check intersection
        numeric_check = LinearModel.check_robust(out_lbs, out_ubs, label)
        if numeric_check:
            return True, None
        else:
            bounds = {
                'matrix': self.layer.weight,
                'offset': self.layer.bias
            }
            return LinearModel.check_sym_robust(in_lbs, in_ubs, bounds, label)
