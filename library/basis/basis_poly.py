import numpy as np
from utils.rl.nmultichoosek import nmultichoosek


def basis_poly(degree, dim, offset, state=None):
    # BASIS_POLY Computes full polynomial features: phi(s) = s^0+s^1+s^2+...
    # Since s is a vector, s^i denotes all the possible products of degree i
    # between all elements of s, e.g.,

    # s = (a, b, c)'
    # s^3 = a^3 + b^3 + c^3 + a^2b + ab^2 + ac^2 + a^2c + b^2c + bc^2

    #    INPUT
    #     - degree : degree of the polynomial
    #     - dim    : dimension of the state
    #     - offset : 1 if you want to include the 0-degree component,
    #                0 otherwise
    #     - state  : (optional) [D x N] matrix of N states of size D to evaluate

    #    OUTPUT
    #     - Phi    : if a state is provided as input, the function
    #                returns the feature vectors representing it;
    #                otherwise it returns the number of features

    # =========================================================================
    # EXAMPLE
    # basis_poly(2,3,1,[3,5,6]') = [1, 3, 5, 6, 9, 15, 18, 25, 30, 36]'

    if state is None:
        dimPhi = nmultichoosek(dim+1, degree, True)
        Phi = dimPhi
        if not offset:
            Phi = Phi - 1
    else:
        assert (state.shape[0] == dim, 'State size is %d. Should be %d.', state.shape[0], dim)
        nSamples = state.shape[1]
        C = nmultichoosek(np.concatenate((np.ones((1, nSamples)), state), axis=0), degree, False)
        Phi = np.prod(C, 1)
        if not offset:
            Phi = np.delete(Phi, 0, 0)

    return Phi