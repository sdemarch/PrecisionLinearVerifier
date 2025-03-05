import matplotlib.pyplot as plt
import numpy as np
import pynever.datasets as dt
import torch
import torchvision.transforms as tr
from mpmath import mp

from linearverifier.core.model import LinearModel


def get_seed(model: LinearModel, idx: int) -> tuple[int, torch.Tensor, bool]:
    """Procedure to load a robust sample from the training set"""

    # Load dataset
    training_set = dt.TorchMNIST('Data',
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


def get_xt(index: int) -> mp.matrix:
    """Get column vector for input idx"""
    with open('jiarinard/binary/XT.csv', 'r') as xt_f:
        line = xt_f.readlines()[index]

        return mp.matrix(line.strip('\n').split(','))


def get_yt(index: int) -> int:
    """Get true prediction for input idx"""
    with open('jiarinard/binary/YT.csv', 'r') as yt_f:
        line = yt_f.readlines()[index]
        return int(line)


def get_yp(index: int) -> mp.mpf:
    """Get prediction for input idx"""
    with open('jiarinard/binary/YP.csv', 'r') as yp_f:
        line = yp_f.readlines()[index]
        return mp.mpf(line)


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


def limiter(x: mp.matrix, low: float = 0, high: float = 1) -> mp.matrix:
    """Limiter for compressing the values of a vector"""
    for i in range(x.rows):
        for j in range(x.cols):
            if x[i, j] <= low:
                x[i, j] = mp.mpf(low)
            elif x[i, j] >= high:
                x[i, j] = mp.mpf(high)

    return x


def norm(x: mp.matrix) -> mp.mpf:
    """Procedure to compute the norm of a vector"""
    return mp.sqrt((x * x.T)[0])


def round_vec(x: mp.matrix, digits: int) -> np.ndarray:
    """Manual rounding"""
    result = np.zeros((x.rows, x.cols))

    for i in range(x.rows):
        for j in range(x.cols):
            result[i, j] = round(float(x[i, j]) * 10 ** digits) / 10 ** digits

    return result


def display(x: mp.matrix) -> None:
    pixels = round_vec(x, 16).reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show(block=False)
    plt.pause(0.001)


def jia_binary(idx: int):
    digits = 3
    PATH = 'jiarinard/binary'

    # Load binary classifier
    mnist = LinearModel(f'{PATH}/w_mnist.csv', f'{PATH}/b_mnist.csv')
    w = mnist.layer.weight

    ###################################
    #       PART 1 - FIND ALPHA       #
    ###################################
    print('******* PARTE 1 *******')

    x_o = get_xt(idx)
    yt = get_yt(idx)
    yp = get_yp(idx)

    lb = 0
    ub = 5
    x_a = None

    # Get robust alpha
    while ub - lb > 1 / 10 ** digits:
        mid = (lb + ub) / 2
        scale = mid * w.T / norm(w)
        candidate_xa = limiter(x_o - scale) if yt > 0 else limiter(x_o + scale)

        robust = check_robust(mnist, candidate_xa, yt)

        if robust:
            lb = mid
            x_a = candidate_xa
        else:
            ub = mid

    # Alpha is the greatest lb such that x_a is robust
    alpha = lb
    y_xa = w * x_a

    # Get alpha limits
    print(f'Y vera       = {yt}')
    print(f'Y predetta   = {yp}')
    print(f'Alpha        = {alpha}')
    print(f'Y_a predetta = {y_xa}')

    ###############################
    #   PART 2 - FIND ADVERSARY   #
    ###############################
    print('******* PARTE 2 *******')

    for i in range(w.cols):
        # Depending on the direction I look at the weights differently
        if w[i] > 0 if yt > 0 else w[i] < 0:
            x_a[i] -= 0.01
            if x_a[i] < 0:
                x_a[i] = 0.0
        else:
            x_a[i] += 0.01
            if x_a[i] > 1:
                x_a[i] = 1.0

    y_adv = w * x_a
    y_adv_np = np.matmul(round_vec(w, digits), round_vec(x_a, digits))[0]
    print(f'Y_adv (normale)  = {y_adv}')
    print(f'Y_adv (troncata) = {y_adv_np}')
    print(f'Y_adv > Y_adv_np? {y_adv > y_adv_np}')


def jia_new(idx: int):
    digits = 3  # MNIST is rounded to the third digit

    # Loads MNIST weights and bias from file
    mnist = LinearModel('Data/MNIST/mnist_weights.txt', 'Data/MNIST/mnist_bias.txt')
    w = mnist.layer.weight
    b = mnist.layer.bias

    # 8576 -> cambia p_oo e rimane p_qq
    # 10   -> rimane p_oo e cambia p_qq
    label, x_o, robust = get_seed(mnist, idx)
    alpha_safe = 0
    display(x_o)

    # TODO: find good values for lb, ub alpha (using binary search?)

    # step = int((ub-lb) * 10**digits)
    # for a in np.linspace(lb, ub, step):
    for a in np.linspace(1, 2, 1000):
        alpha = mp.mpf(a)

        if label > 0:  # TODO CHANGE!
            # Do I still have to compress here? Or just stop when I'm out of the dynamic range?
            # Or (2) try and verify out of the range?
            x_a = x_o - alpha * w[label, :].T / norm(w[label, :])
        else:
            x_a = x_o + alpha * w[label, :].T / norm(w[label, :])

        # Try 1: compress x_a
        x_a = limiter(x_a, 0, 1)

        if check_robust(mnist, x_a, label):
            alpha_safe = alpha

        pred_oo = mnist.layer.predict(x_a)

        with mp.workdps(digits - 1):
            pred_qq = mnist.layer.predict(x_a)

        out_qq_np = np.matmul(round_vec(w, digits), round_vec(x_a, digits)) + round_vec(b, digits)
        pred_qq_np = np.argmax(out_qq_np)

        if pred_oo != pred_qq_np:
            display(x_a)
            print(f'Alpha safe    = {alpha_safe}')
            print(f'Alpha stop    = {alpha}')  # Se alpha == alpha_safe -> attacco riuscito
            print(f'p_oo          = {pred_oo}')
            print(f'p_qq (mpmath) = {pred_qq}')
            print(f'p_qq (numpy)  = {pred_qq_np}')
            print(f'Y_qq (numpy)  = \n{out_qq_np}')
            break
