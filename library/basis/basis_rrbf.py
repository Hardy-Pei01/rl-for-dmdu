import numpy as np


def basis_rrbf(centers, widths, offset, state=None):
    # BASIS_RRBF Uniformly distributed Roooted Gaussian Radial Basis Functions.
    # Phi(i) = exp(-||state - centers(i)|| / widths(i))

    #    INPUT
    #     - n_centers : number of centers (the same for all dimensions)
    #     - widths    : array of widths for each dimension
    #     - range     : [D x 2] matrix with min and max values for the
    #                   D-dimensional input state
    #     - offset    : 1 to add an additional constant of value 1, 0 otherwise
    #     - state     : (optional) [D x N] matrix of N states of size D to
    #                   evaluate

    #    OUTPUT
    #     - Phi       : if a state is provided as input, the function
    #                   returns the feature vectors representing it;
    #                   otherwise it returns the number of features

    # =========================================================================
    # EXAMPLE
    # basis_rrbf(2, [0.3; 0.2], [0 1; 0 1], 0, [0.2; 0.1])
    #     0.4346
    #     0.0663
    #     0.0106
    #     0.0053

    if state is None:
        Phi = centers.shape[1] + (offset == 1)
    else:
        B = 1.0 / (widths ** 2)
        stateB = state * np.sqrt(B)
        centersB = centers * np.sqrt(B)
        if centers.shape[0] > 1:
            stateB = stateB[:, None].T
            centersB = centersB[:, None].T
            Phi = np.exp(-np.sqrt(np.sum((centersB - stateB) ** 2, 2)))
        else:
            centersB = centersB.T
            Phi = np.exp(-np.sqrt((centersB - stateB) ** 2))

        if offset == 1:
            Phi = np.concatenate((np.ones(state.shape[1]), Phi), axis=0)

    return Phi
