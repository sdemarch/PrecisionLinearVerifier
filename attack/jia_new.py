import random

import pynever.datasets as dt
import torch
import torchvision.transforms as tr
from mpmath import mp

from linearverifier.core.model import LinearModel


def get_seed(model: LinearModel) -> tuple[int, torch.Tensor, int]:
    """Procedure to load a robust sample from the training set"""

    # Load dataset
    training_set = dt.TorchMNIST('./classifier',
                                 train=True,
                                 transform=tr.Compose([tr.ToTensor(),
                                                       tr.Lambda(lambda x: torch.flatten(x))]),
                                 download=True)

    # Extract a sample
    robust = False
    t, label, idx = None, 0, 0

    while not robust:
        idx = random.randint(0, len(training_set) - 1)
        t, label = training_set[idx]
        t_mp = mp.matrix(t)
        robust = check_robust(model, t_mp, 1.0, label)
        print(f'Sample {idx} - label {label}, robust = {robust}')

    return idx, t, label


def generate_vnnlib(x: mp.matrix, scale: float, label: int, name: str = 'jia_prop.vnnlib',
                    eps: float = 0.01) -> str:
    """Generate the VNNLIB property for solvers"""

    with open(name, 'w') as p:
        # Variables
        for i in range(x.rows):
            p.write(f'(declare-const X_{i} Real)\n')
        p.write('\n')
        for i in range(10):
            p.write(f'(declare-const Y_{i} Real)\n')
        p.write('\n')

        # Constraints on X
        for i in range(x.rows):
            # Property is within the dynamic range [0, 1]
            p.write(f'(assert (>= X_{i} {max(scale * x[i] - eps, mp.mpf(0))}))\n')
            p.write(f'(assert (<= X_{i} {min(scale * x[i] + eps, mp.mpf(1))}))\n')
        p.write('\n')

        # Constraints on Y
        p.write('(assert (or\n')
        for i in range(10):
            if i != label:
                p.write(f'\t(<= Y_{label} Y_{i})\n')
        p.write('))')

    return name


def check_robust(model: LinearModel, x: mp.matrix, scale: float | mp.mpf, label: int, eps: float = 0.01) -> bool:
    """Procedure to call the solver to check robustness"""

    return model.verify(generate_vnnlib(x, scale, label, name=f'jia_prop_label_{label}.vnnlib', eps=eps))


def norm(x: mp.matrix) -> mp.matrix:
    """Procedure to compute the norm of a vector"""
    return 1

digits = 3  # MNIST is rounded to the third digit

# Loads MNIST weights and bias from file
mnist = LinearModel('../Data/MNIST/mnist_weights.txt', '../Data/MNIST/mnist_bias.txt')
w = mnist.layer.weight
b = mnist.layer.bias

_, x_o, label = get_seed(mnist)
x_o = mp.matrix(x_o)

for alpha in range(0, 1, 1000):
    if label > 0:  # TODO CHANGE!
        x_a = x_o - alpha * (w[label, :].T / norm(w[label, :]))

print(mnist.layer.weight[2, 5])

with mp.workdps(digits - 1):
    w_q = mnist.layer.weight
    print(w_q[2, 5])

print(w_q[2, 5])
print((round(w_q[2, 5] * 10 ** digits)) / 10 ** digits)
