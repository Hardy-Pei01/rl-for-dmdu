import numpy as np
from utils.rl.pareto import pareto


def metric_hv(J=None, HVF=None):
    # METRIC_HV Scalarizes a set of solution for multi-objective problem (i.e.,
    # an approximate Pareto frontier). A solution is ranked according to its
    # contribution to the hypervolume of the frontier.

    #    INPUT
    #     - J   : [N x M] matrix of samples to evaluate, where N is the number
    #             of samples and M the number of objectives
    #     - HVF : hypervolume function handle

    #    OUTPUT
    #     - S   : [N x 1] vector of the hypervolume contribution of each sample

    # =========================================================================
    # REFERENCE
    # N Beume, B Naujoks, M Emmerich
    # SMS-EMOA: Multiobjective selection based on dominated hypervolume (2007)

    uniqueJ, _, idx = np.unique(J, return_index=True, return_inverse=True, axis=0)

    front, _, _, idx2 = pareto(uniqueJ)

    hyperv_ref = HVF(front)

    hvContrib = []
    for i in range(0, front.shape[0]):
        front_tmp = front
        front_tmp = np.delete(front_tmp, i, axis=0)
        hvContrib.append(hyperv_ref - HVF(pareto(front_tmp)[0]))

    hvContrib = np.array(hvContrib)
    hvUnique = np.zeros((uniqueJ.shape[0], 1))
    for i in range(len(idx2)):
        hvUnique[idx2[i]] = hvContrib[i]
    S = hvUnique[idx]

    S = np.maximum(0, S)

    S[S == 0] = - 0.001

    return S
