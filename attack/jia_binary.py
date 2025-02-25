import numpy as np
from mpmath import mp

digits = 3
idx = 1


def get_xt(index: int) -> mp.matrix:
    """Get column vector for input idx"""
    with open('binary/XT.csv', 'r') as xt_f:
        line = xt_f.readlines()[index]

        return mp.matrix(line.strip('\n').split(','))


def get_yt(index: int) -> float:
    """Get true prediction for input idx"""
    with open('binary/YT.csv', 'r') as yt_f:
        line = yt_f.readlines()[index]
        return float(line)


def get_yp(index: int) -> mp.mpf:
    """Get prediction for input idx"""
    with open('binary/YP.csv', 'r') as yp_f:
        line = yp_f.readlines()[index]
        return mp.mpf(line)

def round_vec(x: mp.matrix, force_round = True) -> np.ndarray:
    """Manual rounding"""
    result = np.array(x)

    for i in range(max(x.rows, x.cols)):
        if force_round:
            result[i] = round(float(result[i]) * 10**digits) / 10**digits
        else:
            result[i] = (result[i] * 10**digits) / 10**digits

    return result



with open('binary/w_mnist.csv', 'r') as f:
    weights = [float(l) for l in f]
    w = mp.matrix(weights)

########################################
### PARTE 1 - STABILIRE LIMITI ALPHA ###
########################################
print('******* PARTE 1 *******')

x_o = get_xt(idx)
yt = get_yt(idx)
yp = get_yp(idx)
alpha = 2

print(f'Y vera      = {yt}')
print(f'Y predetta  = {yp}')
print(f'Alpha       = {alpha}')

wn = mp.sqrt((w.T * w)[0])

if yt > 0:
    x_a = x_o - alpha * w / wn
else:
    x_a = x_o + alpha * w / wn

print(f'Y_oo    = {x_a.T * w}')
with mp.workdps(digits - 1):
    print(f'Y_qq    = {x_a.T * w}')

##############################
### PARTE 2 - TROVARE BUCO ###
##############################
print('******* PARTE 2 *******')

for alpha in np.linspace(1, 2, 1000):
    a = mp.mpf(alpha)

    if yt > 0:
        x_a = x_o - a * w / wn
    else:
        x_a = x_o + a * w / wn

    yoo = (x_a.T * w)[0]
    yqq_np = np.matmul(round_vec(x_a, False), round_vec(w))

    if yqq_np * yoo < 0:
        print(f'Alpha         = {alpha}')
        print(f'Y_oo          = {yoo}')
        with mp.workdps(digits - 1):
            print(f'Y_qq (mpmath) = {(x_a.T * w)[0]}')
        print(f'Y_qq (numpy)  = {yqq_np}')
