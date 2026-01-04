import math
from ast import literal_eval

import numpy as np

def get_input_data():
    with open('data/RandomNumbers.txt') as f:
        data = f.read()
    return data


def array_to_matrix(input_numbers):
    n = int(math.isqrt(len(input_numbers)))
    if n * n != len(input_numbers):
        return False, False

    # Sort descending
    input_numbers = sorted(input_numbers, reverse=True)

    # Build rotated masks for diagonals
    masks = [np.rot90(np.eye(n, k=k), k=-1) for k in range(n - 1, -n, -1)]
    idxs = [np.where(m) for m in masks]
    r, c = zip(*idxs)

    mat = np.zeros((n, n), dtype=int)
    for ri, ci in zip(r, c):
        for rr, cc in zip(ri.tolist(), ci.tolist()):
            mat[rr, cc] = input_numbers.pop()

    # Same monotonicity check
    checked = (np.all(np.diff(mat, axis=0) < 0) and
               np.all(np.diff(mat, axis=1) < 0))

    return checked, mat.tolist()

if __name__ == "__main__":
    total = 0
    input_arrays = get_input_data().split('\n')
    for arr in input_arrays:
        arr = literal_eval(arr)
        c, d = array_to_matrix(arr)
        total += int(c)
        if c:
            print(d)
        else:
            print('Failed')
    print(total)