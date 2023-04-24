import numpy as np


def basis_krbf(centers, B, offset, state):
    # BASIS_KRBF Kernel Radial Basis Functions.
    # Phi = exp( -(state - centers)' * B * (state - centers) ),
    # where B is a diagonal matrix denoting the bandwiths of the kernels.
    # Centers are uniformly placed in RANGE and bandwidths are automatically
    # computed. See the code for more details.

    #    INPUT
    #     - n_centers : number of centers (the same for all dimensions)
    #     - range     : [D x 2] matrix with min and max values for the
    #                   D-dimensional input state
    #     - offset    : 1 to add an additional constant of value 1, 0 otherwise
    #     - state     : (optional) [D x N] matrix of N states of size D to
    #                   evaluate

    #    OUTPUT
    #     - Phi       : if a state is provided as input, the function
    #                   returns the feature vectors representing it;
    #                   otherwise it returns the number of features

    if state is None:
        Phi = centers.shape[1] + 1 * (offset == 1)
    else:
        stateB = state * np.sqrt(B)
        centersB = centers * np.sqrt(B)
        stateB = np.transpose(stateB[:, :, None], (2, 1, 0))
        centersB = np.transpose(centersB[:, :, None], (1, 2, 0))
        Phi = np.exp(np.sum(-(centersB - stateB) ** 2, 2))

    return Phi
