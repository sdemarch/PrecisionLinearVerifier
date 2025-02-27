import numpy as np
import pynever.datasets as dt
import torch
import torchvision.transforms as tr
from mpmath import mp

from linearverifier.core.model import LinearModel


def get_seed(model: LinearModel, idx: int) -> tuple[int, torch.Tensor, bool]:
    """Procedure to load a robust sample from the training set"""

    # Load dataset
    training_set = dt.TorchMNIST('./classifier',
                                 train=True,
                                 transform=tr.Compose([tr.ToTensor(),
                                                       tr.Lambda(lambda x: torch.flatten(x))]),
                                 download=True)

    # Extract a sample
    t, label = training_set[idx]
    t_mp = mp.matrix(t)
    robust = check_robust(model, t_mp, label)
    print(f'Sample {idx} - label {label}, robust = {robust}')

    return label, t_mp, robust


def generate_vnnlib(x: mp.matrix, label: int, name: str = 'jia_prop.vnnlib', eps: float = 0.01) -> str:
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
            p.write(f'(assert (>= X_{i} {max(x[i] - eps, mp.mpf(0))}))\n')
            p.write(f'(assert (<= X_{i} {min(x[i] + eps, mp.mpf(1))}))\n')
        p.write('\n')

        # Constraints on Y
        p.write('(assert (or\n')
        for i in range(10):
            if i != label:
                p.write(f'\t(<= Y_{label} Y_{i})\n')
        p.write('))')

    return name


def check_robust(model: LinearModel, x: mp.matrix, label: int, eps: float = 0.01) -> bool:
    """Procedure to call the solver to check robustness"""

    return model.verify(generate_vnnlib(x, label, name=f'jia_prop_label_{label}.vnnlib', eps=eps))


def norm(x: mp.matrix) -> mp.mpf:
    """Procedure to compute the norm of a vector"""
    return mp.sqrt((x * x.T)[0])


def round_vec(x: mp.matrix, force_round=True) -> np.ndarray:
    """Manual rounding"""
    result = np.zeros((x.rows, x.cols))

    for i in range(x.rows):
        for j in range(x.cols):
            if force_round:
                result[i, j] = round(float(x[i, j]) * 10 ** digits) / 10 ** digits
            else:
                result[i, j] = (float(x[i, j]) * 10 ** digits) / 10 ** digits

    return result


digits = 3  # MNIST is rounded to the third digit

# Loads MNIST weights and bias from file
mnist = LinearModel('../Data/MNIST/mnist_weights.txt', '../Data/MNIST/mnist_bias.txt')
w = mnist.layer.weight
b = mnist.layer.bias

# 8576
label, x_o, robust = get_seed(mnist, 8576)
alpha_safe = 0
counter = 0

for a in np.linspace(1, 2, 1000):
    alpha = mp.mpf(a)
    print(a)

    if label > 0:  # TODO CHANGE!
        x_a = x_o - alpha * w[label, :].T / norm(w[label, :])
    else:
        x_a = x_o + alpha * w[label, :].T / norm(w[label, :])

    if not check_robust(mnist, x_a, label):
        print(alpha_safe)
    else:
        alpha_safe = alpha

    pred_oo = mnist.layer.predict(x_a)
    counter += 1
    if pred_oo != label:
        check_robust(mnist, x_a, label)

    with mp.workdps(digits - 1):
        pred_qq = mnist.layer.predict(x_a)

    out_qq_np = np.matmul(round_vec(w), round_vec(x_a, False)) + round_vec(b)
    pred_qq_np = np.argmax(out_qq_np)

    if pred_oo != pred_qq_np:
        print(f'p_oo          = {pred_oo}')
        print(f'p_qq (mpmath) = {pred_qq}')
        print(f'p_qq (numpy)  = {pred_qq_np}')
        print(f'Y_qq (numpy)  = {out_qq_np}')
        print('-----------------')
