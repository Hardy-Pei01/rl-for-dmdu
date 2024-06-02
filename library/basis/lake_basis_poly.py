import numpy as np


def lake_basis_poly(state=None):
    numfeatures = 5
    # If no arguments just return the number of basis functions
    if state is None:
        phi = numfeatures
        return phi

    d, n = state.shape
    phi = np.zeros((numfeatures, n))
    phi[0, :] = state[0, :] # pollution
    phi[1, :] = state[1, :] # utility
    phi[2, :] = state[2, :] # reliability
    # if eutrophication, set 4th to 10 * pollution
    # otherwise, set 4th to pollution
    phi[3, phi[2, :] != 0] = phi[0, phi[2, :] != 0]
    phi[3, phi[2, :] == 0] = phi[0, phi[2, :] == 0] * 10
    # if eutrophication, set 5th to 0
    # otherwise, set 5th to pollution
    phi[4, phi[2, :] != 0] = phi[0, phi[2, :] != 0]
    # phi[5, :] = (state[0, :] == 0)
    # phi[6, :] = (state[1, :] == 0)
    # phi[7, :] = (state[2, :] == 0)
    return phi
