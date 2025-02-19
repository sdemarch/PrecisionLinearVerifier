"""
This module reads a VNNLIB file containing a robustness specification

"""

from mpmath import mp


def read_vnnlib(filename: str) -> tuple[mp.matrix, mp.matrix, int]:
    lbs = []
    ubs = []
    label = int(filename.split('_')[-1].replace('.vnnlib', ''))

    with open(filename, 'r') as vnnlib_file:
        # Input condition
        for line in vnnlib_file:
            if '>=' in line:
                lbs.append(mp.mpf(line.split()[-1].replace('))', '')))
            elif '<=' in line:
                ubs.append(mp.mpf(line.split()[-1].replace('))', '')))
            elif 'or' in line:
                # We reached the output condition
                break

    return mp.matrix(lbs), mp.matrix(ubs), label
