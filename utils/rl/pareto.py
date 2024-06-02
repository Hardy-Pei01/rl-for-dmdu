import numpy as np


def pareto(s=None, p=None, std=None):
    # PARETO Filters a set of points S according to Pareto dominance, i.e.,
    # points that are dominated (both weakly and strongly) are filtered.

    #    INPUT
    #     - S    : [N x D] matrix, where N is the number of points and D is the
    #              number of elements (objectives) of each point.
    #     - P    : (optional) [N x T] matrix containing the policies/parameters
    #              that generated S

    #    OUTPUT
    #     - S    : Pareto-filtered S
    #     - P    : Pareto-filtered P
    #     - idxs : indices of the non-dominated solutions

    # =========================================================================
    # EXAMPLE
    # s = [1 1 1; 2 0 1; 2 -1 1; 1, 1, 0];
    # [f, ~, idxs] = pareto(s)
    #     f = [1 1 1; 2 0 1]
    #     idxs = [1; 2]

    i, dim = s.shape
    if p is None:
        p = np.zeros((i, 1))
    if std is None:
        std = np.zeros((i, 1))

    idxs = np.arange(0, i)
    while i >= 1:
        idx = i - 1
        old_size = s.shape[0]
        indices = np.sum((s[idx, :] >= s), 1) == dim
        indices[idx] = False
        s = np.delete(s, indices == True, axis=0)
        p = np.delete(p, indices == True, axis=0)
        std = np.delete(std, indices == True, axis=0)
        idxs = np.delete(idxs, indices == True, axis=0)
        i = i - 1 - (old_size - s.shape[0]) + np.sum(indices[idx:])

    return s, p, std, idxs
