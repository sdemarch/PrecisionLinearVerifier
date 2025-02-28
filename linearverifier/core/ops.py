"""
This module defines common operations for the verification algorithm
"""

from mpmath import mp


def get_positive(a: mp.matrix) -> mp.matrix:
    """Procedure to extract the positive part of a matrix"""
    result = mp.matrix(a.rows, a.cols)

    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] if a[i, j] > 0 else mp.mpf(0)

    return result


def get_negative(a: mp.matrix) -> mp.matrix:
    """Procedure to extract the negative part of a matrix"""
    result = mp.matrix(a.rows, a.cols)

    for i in range(a.rows):
        for j in range(a.cols):
            result[i, j] = a[i, j] if a[i, j] < 0 else mp.mpf(0)

    return result


def create_disjunction_matrix(n_outs: int, label: int) -> list[mp.matrix]:
    """Procedure to create the matrix of the output property"""
    matrix = []
    c = 0

    if n_outs == 1:
        result = [mp.matrix(1, 1)]
        result[0][0] = 1 if label == 1 else -1
        return result

    for i in range(n_outs):
        if i != label:
            matrix.append(mp.matrix(1, n_outs))
            matrix[c][label] = 1
            matrix[c][i] = -1
            c += 1

    return matrix


def check_unsafe(bounds: dict, matrix: mp.matrix) -> bool:
    """Procedure to check whether the output bounds are unsafe"""

    m_plus = get_positive(matrix)
    m_minus = get_negative(matrix)
    lbs = bounds['lower']
    ubs = bounds['upper']

    min_value = m_plus * lbs + m_minus * ubs

    assert min_value.rows == min_value.cols == 1

    # Since we're linear we know for sure this is enough
    if min_value[0] > 0:
        return False
    else:
        return True


def check_unsafe_sym(bounds: dict, matrix: mp.matrix, lbs: mp.matrix, ubs: mp.matrix) -> bool:
    """Procedure to check whether the output symbolic bounds are unsafe"""

    # Assumption: the lower and upper bounds are the same (1 dense layer)
    out_m = get_positive(matrix) * bounds['matrix'] + get_negative(matrix) * bounds['matrix']
    out_b = get_positive(matrix) * bounds['offset'] + get_negative(matrix) * bounds['offset']

    out_m_plus = get_positive(out_m)
    out_m_minus = get_negative(out_m)

    min_value = out_m_plus * lbs + out_m_minus * ubs + out_b

    assert min_value.rows == min_value.cols == 1

    if min_value[0] > 0:
        return False
    else:
        return True
