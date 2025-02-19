import copy
import random
from argparse import ArgumentParser

import numpy as np
import pynever.datasets as dt
import torch
from mpmath import mp
from torchvision import transforms as tr

from linearverifier.core.model import LinearModel

parser = ArgumentParser(description='Jia-Rinard method evaluation')

parser.add_argument('nn', type=str, help='Path to neural network model')
parser.add_argument('-e', '--eps_r', type=float, default=1e-7, help='Epsilon value that guides the search'
                                                                    ' of the coefficient alpha to reach the decision '
                                                                    'boundary')
parser.add_argument('-l', '--l_inf', type=float, default=0.01, help='l-inf perturbation to check '
                                                                    'robustness')
parser.add_argument('-p', '--precision', type=int, default=4, help='mp.dps precision of NeVer')
parser.add_argument('-n', '--n_samples', type=int, default=1, help='Number of samples to test')

args = parser.parse_args()

EPS_r = args.eps_r  # 1e-7 for 32-bit precision
EPS_linf = args.l_inf  # 0.03 / 0.01


def softmax(x):
    """Compute softmax values for each sets of scores in x"""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_seed(nn: str) -> tuple[int, torch.Tensor, int]:
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
        robust = check_robust(nn, t_mp, 1.0, label)
        print(f'Sample {idx} - label {label}, robust = {robust}')

    return idx, t, label


def generate_vnnlib(x: mp.matrix, scale: float, label: int, name: str = 'jia_prop.vnnlib') -> str:
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
            p.write(f'(assert (>= X_{i} {max(scale * x[i] - EPS_linf, mp.mpf(0))}))\n')
            p.write(f'(assert (<= X_{i} {min(scale * x[i] + EPS_linf, mp.mpf(1))}))\n')
        p.write('\n')

        # Constraints on Y
        p.write('(assert (or\n')
        for i in range(10):
            if i != label:
                p.write(f'\t(<= Y_{label} Y_{i})\n')
        p.write('))')

    return name


def check_robust(nn: str, x: mp.matrix, scale: float, label: int) -> bool:
    """Procedure to call the solver to check robustness"""

    return LinearModel(nn).verify(generate_vnnlib(x, scale, label, name=f'jia_prop_{label}.vnnlib'))


def find_x0(x: torch.Tensor, label: int, nn: str) -> tuple[float, torch.Tensor]:
    """Procedure to find x_0 = alpha*x using binary search to find alpha"""

    # Initial search bounds (1 * x_seed is by definition safe
    # and 0 * x_seed is a total black image)
    a1 = 0
    a = 1

    # Truncate x
    x_t = mp.matrix(x)
    loop = 0

    # Stop when NN_a is safe and NN_a1 is unsafe and a - a1 < EPS_r
    while a - a1 > EPS_r:
        # Scale the original image
        mid = (a + a1) / 2

        # Binary search to converge
        robust = check_robust(nn, x_t, mid, label)
        if robust:
            a = mid
        else:
            a1 = mid

        # Log
        print('-----------------------')
        print(f'loop: {loop}, alpha = {a}, alpha1 = {a1}, (alpha - alpha1) = {a - a1}, robust: {robust}')
        print(f'Prediction with alpha: {predict(nn, a * x)}')
        print(f'Prediction with alpha - truncated: {predict_trunc(nn, a * x_t)}')
        print('-')
        print(f'Prediction with alpha1: {predict(nn, a1 * x)}')
        print(f'Prediction with alpha1 - truncated: {predict_trunc(nn, a1 * x_t)}')
        loop += 1

    return a, a * x  # x_0 = x * alpha


def find_adv(x: torch.Tensor, label: int, nn: str) -> torch.Tensor:
    # Initialize x_adv as x_0
    adv = copy.deepcopy(x)

    W = LinearModel(nn).layer.weight

    # Scale the pixels to reach the frontier
    for i in range(len(adv)):
        if W[label, i] > 0:
            adv[i] -= EPS_linf
            if adv[i] < 0:
                adv[i] = 0.0

        else:
            adv[i] += EPS_linf
            if adv[i] > 1:
                adv[i] = 1.0

    return adv


def check_in_radius(x: torch.Tensor | mp.matrix, x0: torch.Tensor) -> bool:
    """Procedure to check whether the Tensor x belongs to the l_inf norm of x0"""

    if isinstance(x, torch.Tensor):
        for i in range(len(x)):
            if x[i] > x0[i] + EPS_linf or x[i] < x0[i] - EPS_linf:
                return False
    else:
        for i in range(len(x)):
            if mp.mpf(x[i]) > mp.mpf(x0[i].item()) + EPS_linf or mp.mpf(x[i]) < mp.mpf(x0[i].item()) - EPS_linf:
                return False

    return True


def predict(onnx: str, x_in: torch.Tensor) -> int:
    import pynever.strategies.conversion.representation as cv
    import pynever.strategies.conversion.converters.onnx as ox
    from pynever.utilities import execute_network

    nn = ox.ONNXConverter().to_neural_network(cv.load_network_path(onnx))
    return np.argmax(softmax(execute_network(nn, x_in)))


def predict_trunc(nn: str, x_in: mp.matrix) -> int:
    # Get output
    fc = LinearModel(nn).layer
    out_layer = fc.weight * x_in + fc.bias
    return np.argmax(out_layer)


def save_tensor(x: torch.Tensor, idx: int, label: int, root_dir: str = 'Experiments') -> None:
    """Save the Tensor x to txt file"""
    with open(f'{root_dir}/adv_{idx}_label_{label}.txt', 'w') as f:
        for v in x:
            f.write(str(float(v)) + '\n')


def save_property(x: torch.Tensor, idx: int, label: int, root_dir: str = 'Experiments') -> None:
    """Save the property to verify"""
    with open(f'{root_dir}/prop_{idx}_label_{label}.txt', 'w') as p:
        # Variables
        for i in range(len(x)):
            p.write(f'(declare-const X_{i} Real)\n')
        p.write('\n')
        for i in range(10):
            p.write(f'(declare-const Y_{i} Real)\n')
        p.write('\n')

        # Constraints on X
        for i in range(len(x)):
            # Property is within the dynamic range [0, 1]
            p.write(f'(assert (>= X_{i} {max(x[i] - EPS_linf, 0.0)}))\n')
            p.write(f'(assert (<= X_{i} {min(x[i] + EPS_linf, 1.0)}))\n')
        p.write('\n')

        # Constraints on Y
        p.write('(assert (or\n')
        for i in range(10):
            if i != label:
                p.write(f'\t(and (<= Y_{label} Y_{i}))\n')
        p.write('))')


def main():
    with open('results.csv', 'w') as out:

        out.write('Sample,Precision epsilon,l_inf noise,Original label,Coefficient alpha,'
                  'Adversary label,Truncated adversary label,Success\n')
        for _ in range(args.n_samples):

            # Step 0 - Get x_seed which is robust for the loaded network
            sample, x_seed, pred = get_seed(args.nn)

            # Step 1 - Find x_0 based on x_seed
            alpha, x_0 = find_x0(x_seed, pred, args.nn)
            pred_x0 = predict(args.nn, x_0)

            print('***********************************')
            print(f'Original label of x_seed: {pred}')
            print(f'Prediction on x_seed: {predict(args.nn, x_seed)}')
            print(f'alpha = {alpha}')
            print(f'Prediction on x_0 = alpha * x_seed: {pred_x0}')

            if predict(args.nn, x_seed) != pred:
                with open(f'exception_sample_{sample}.txt') as e:
                    e.write(f'P(xseed) = {pred}\nP(x0) = {pred_x0}\n')

            # Step 2 - Find x_adv where x_0 was deemed safe (both precise and truncated)
            x_adv = find_adv(x_0, pred, args.nn)
            pred_adv = predict(args.nn, x_adv)

            x_adv_t = mp.matrix(x_adv)
            pred_adv_t = predict_trunc(args.nn, x_adv_t)

            # Step 3 - If the low-precision implementation has an adversary save it
            if pred_adv_t != pred_adv:
                save_property(x_0, sample, pred)
                save_tensor(x_adv, sample, pred)

            print(f'Prediction on x_adv: {pred_adv}')
            print(f'Prediction on x_adv - truncated: {pred_adv_t}')
            print(f'x_adv belongs to Adv(x_0): {check_in_radius(x_adv, x_0)}')
            print(f'x_adv_t belongs to Adv(x_0): {check_in_radius(x_adv_t, x_0)}')
            success = check_in_radius(x_adv, x_0) and check_in_radius(x_adv_t, x_0) and pred_adv_t != pred_adv

            out.write('{},{},{},{},{},{},{},{}\n'.format(sample, EPS_r, EPS_linf, pred, alpha, pred_adv,
                                                         pred_adv_t, success))


if __name__ == "__main__":
    with mp.workdps(args.precision):
        main()
