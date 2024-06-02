import numpy as np


def deep_basis_poly(state=None):
    numfeatures = 5
    # If no arguments just return the number of basis functions
    if state is None:
        phi = numfeatures
        return phi

    d, n = state.shape
    assert (d == 2)
    phi = np.zeros((numfeatures, n))
    phi[0, :] = state[0, :] + 1
    phi[1, :] = state[1, :] + 1
    phi[2, :] = np.multiply((state[0, :]+1), (state[1, :]+1))
    phi[3, :] = np.logical_and(state[0, :] == 0, state[1, :] == 0)
    phi[4, :] = (state[0, :] == 0)
    return phi
