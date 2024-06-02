import itertools
import numpy as np
from scipy.special import comb


def nmultichoosek(values, k, count):
    # NMULTICHOOSEK Like nchoosek, but with repetitions. The VALUES for which
    # nchoosek is performed are columns. If VALUES is a matrix, nchoosek is
    # performed for each column and COMBS is a matrix as well.

    if count:
        d = values
        combs = comb(d + k - 1, k, exact=True)
    else:
        d, ncombs = values.shape
        if k == 1:
            combs = np.arange(0, d+k-1)
        elif k == 2:
            combs = np.array(list(itertools.combinations(np.arange(0, d+k-1), 2)))
            combs[:, 0] = combs[:, 0] - 1
            combs[:, 1] = combs[:, 1] - 2
        else:
            raise NotImplementedError
        combs = values[combs, :].reshape(-1, k, ncombs)

    return combs
