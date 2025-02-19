"""
This is the entry point of the verifier
Usage: python verifier.py NETWORK.onnx PROPERTY.vnnlib

@author Stefano Demarchi

"""

from argparse import ArgumentParser

from mpmath import mp

from linearverifier.core.model import LinearModel

parser = ArgumentParser(description="Linear neural networks verifier")
parser.add_argument('net', type=str, help='Network to test (test or mnist)')
parser.add_argument('prop', type=str, help='VNNLIB property file')
parser.add_argument('--precision', type=int, default=15, help='dps precision for mpmath')

args = parser.parse_args()

if __name__ == '__main__':
    with mp.workdps(args.precision):
        w_path = 'Data/Test/test_weights.txt' if args.net == 'test' else 'Data/MNIST/mnist_weights.txt'
        b_path = 'Data/Test/test_bias.txt' if args.net == 'test' else 'Data/MNIST/mnist_bias.txt'
        model = LinearModel(w_path, b_path)
        result = model.verify(args.prop)

    print(f'Property verified: {result}')
