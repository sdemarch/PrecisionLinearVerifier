"""
This module reads ONNX files representing neural networks
"""

import numpy as np
import onnx

from linearverifier.core.layer import LinearLayer, Layer


def read_weights_from_file(weights_file: str) -> list[float] | list[list[float]]:
    """Procedure to read a file containing weights"""

    with open(weights_file, 'r') as f:
        lines = f.readlines()
        if ',' in lines[0]:
            result = [[float(v) for v in line.split(',')] for line in lines]
        else:
            result = [float(line) for line in lines]

    return result


def nn_from_weights(w_file: str, b_file: str) -> list[Layer]:
    """Procedure to read a file containing weights"""

    # Create layers
    net = []

    weights = read_weights_from_file(w_file)
    bias = read_weights_from_file(b_file)

    net.append(LinearLayer(weights, bias))
    return net


def nn_from_onnx(onnx_path: str) -> list[Layer]:
    """Procedure to read a ONNX network as a list of Layer objects"""

    # Open model
    onnx_net = onnx.load(onnx_path)

    # Read data
    parameters = {}
    for initializer in onnx_net.graph.initializer:
        parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # Create layers
    net = []

    for node in onnx_net.graph.node:
        if node.op_type == 'Gemm':

            weight = parameters[node.input[1]]
            for att in node.attribute:
                if (att.name == 'transA' or att.name == 'transB') and att.i == 0:
                    weight = parameters[node.input[1]].T

            neurons = weight.shape[0]

            bias = np.zeros((neurons, 1))
            if len(node.input) > 2:
                bias = parameters[node.input[2]]

            net.append(LinearLayer(weight.tolist(), bias.tolist()))

    assert (len(net)) == 1
    return net
