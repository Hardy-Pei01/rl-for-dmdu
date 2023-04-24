import numpy as np


def mymnrnd(p=None, n=None):
    # MYMNRND Draws N samples from a multimonial distribution of probabilities
    # P. Either P is a vector of size M or a matrix of size [M x N], where M is
    # the number of variables of the distribution. In the former case, N
    # samples are drawn from the same distribution P. In the latter, N samples
    # are drawn from each of the P distributions (one for each column). In both
    # cases, the random values R will be stored in a row vector.

    # =========================================================================
    # EXAMPLE
    # p = [0.1 0.2 0.3 0.4];
    # n = 10;
    # mymnrnd(p, n) will draw ten integers between [1, 4]

    if len(p.shape) == 1:
        p = np.transpose(p)
    elif len(p.shape) == 2 and p.shape[1] == 1:
        pass
    elif p.shape[1] != n:
        raise Exception('p has wrong dimensions.')

    m = p.shape[0]
    idx0 = np.sum(p, 0) < 1e-20
    a = p[:, idx0 == 0]
    p[:, idx0 == 0] = p[:, idx0 == 0] * (1 / np.sum(p[:, idx0 == 0], 0))

    p[:, idx0] = 1 / m
    F = p.cumsum(axis=0)
    temp = F >= np.random.rand(1, n)
    r = m - np.sum(temp, 0)
    return r
