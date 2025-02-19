"""
This is the entry point of the verifier
Usage: python verifier.py NETWORK.onnx PROPERTY.vnnlib

@author Stefano Demarchi

"""

from argparse import ArgumentParser

from mpmath import mp

from linearverifier.core.model import LinearModel

parser = ArgumentParser(description="Linear neural networks verifier")
parser.add_argument('net', type=str, help='ONNX model file')
parser.add_argument('prop', type=str, help='VNNLIB property file')
parser.add_argument('--precision', type=int, default=15, help='dps precision for mpmath')

args = parser.parse_args()

if __name__ == '__main__':
    with mp.workdps(args.precision):
        model = LinearModel(args.net, 'Data/test_weights.txt', 'Data/test_bias.txt')
        result = model.verify(args.prop)

    print(f'Property verified: {result}')
