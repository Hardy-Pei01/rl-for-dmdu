import numpy as np


def fishing_basis_poly(state=None):
    numfeatures = 7
    # If no arguments just return the number of basis functions
    if state is None:
        phi = numfeatures
        return phi

    d, n = state.shape
    phi = np.zeros((numfeatures, n))
    phi[0, :] = state[0, :] # pollution
    phi[1, :] = state[1, :] # utility
    phi[2, :] = (state[0, :] == 0)
    phi[3, :] = (state[1, :] == 0)
    phi[4, :] = (state[0, :] == 1)
    phi[5, :] = (state[1, :] == 1)
    phi[6, :] = state[0, :] + state[1, :]
    return phi
