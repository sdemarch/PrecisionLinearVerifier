"""
This is the entry point of the verifier
Usage: python verifier.py NETWORK.onnx PROPERTY.vnnlib

@author Stefano Demarchi

"""
import time
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
        model = LinearModel(args.net)

    start = time.perf_counter()
    print(f'Property verified: {model.verify(args.prop)}')
    print(f'Elapsed time     : {time.perf_counter() - start:.4f}s')
