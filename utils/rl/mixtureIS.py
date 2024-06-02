import numpy as np
import numpy.matlib


def mixtureIS(target=None, samplers=None, N=None, Actions=None):
    # MIXTUREIS Computes importance sampling weights. N_TOT samples are drawn
    # from a uniform mixture of N_TOT / N policies. Each policy collects N
    # batches of samples.

    #    INPUT
    #     - target   : the distribution to be updated
    #     - samplers : the distributions used for sampling, assumed to belong
    #                  to an uniform mixture of N_TOT / N policies
    #     - N        : number of samples drawn by each sampler distribution
    #     - Actions  : [D x N_TOT] matrix of parameters (D is the size of the
    #                  parameters)
    #     - States   : (optional) [S x N_TOT] matrix of states (S is the size
    #                  of the state)

    #    OUTPUT
    #     - W        : [1 x N_TOT] importance sampling weights

    # =========================================================================
    # REFERENCE
    # A Owen and Y Zhou
    # Safe and effective importance sampling (2000)

    N_TOT = Actions.shape[1]
    Q = np.zeros((N_TOT, N_TOT))

    p = target.evaluate(Actions)
    for j in range(0, N_TOT, N):
        L = samplers[j].evaluate(Actions).T
        Q[:, j:j+N] = np.tile(L, (N, 1)).T

    Q = Q / N_TOT
    W = p / np.sum(Q, 1)

    W = np.transpose(W)
    return W
