import numpy as np


def fishing3_basis_poly(state=None):
    numfeatures = 10
    # If no arguments just return the number of basis functions
    if state is None:
        phi = numfeatures
        return phi

    d, n = state.shape
    phi = np.zeros((numfeatures, n))
    phi[0, :] = state[0, :] # pollution
    phi[1, :] = state[1, :] # utility
    phi[2, :] = state[2, :] # utility
    phi[3, :] = (state[0, :] == 0)
    phi[4, :] = (state[1, :] == 0)
    phi[5, :] = (state[2, :] == 0)
    phi[6, :] = (state[0, :] == 1)
    phi[7, :] = (state[1, :] == 1)
    phi[8, :] = (state[2, :] == 1)
    # phi[9, :] = state[0, :] + state[1, :]
    # phi[10, :] = state[1, :] + state[2, :]
    # phi[11, :] = state[0, :] + state[2, :]
    phi[9, :] = state[0, :] + state[1, :] + state[2, :]
    return phi
