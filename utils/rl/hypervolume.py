import numpy as np


def hypervolume(F=None, AU=None, U=None, N=None):
    # HYPERVOLUME Approximates the hypervolume of a Pareto frontier. First, it
    # generates random samples in the hypercuboid defined by the utopia and
    # antiutopia points. Second, it counts the number of samples dominated by
    # the frontier.  The hypervolume is approximated as the ratio
    # 'dominated points / total points'.
    # Please notice that the choice of the utopia and antiutopia points is
    # crucial: using points very far from the frontier will result in similar
    # hypervolume even for very different frontiers (if the utopia is too far
    # away, the hypervolume will be always low; if the antiutopia is too far
    # away, the hypervolume will be always high).
    # Also, frontier points "beyond" the reference points will not be counted
    # for the approximation (e.g., if the antiutopia is above the frontier or
    # the utopia is below, the hypervolume will be 0).

    #    INPUT
    #     - F  : the Pareto frontier to evaluate
    #     - AU : antiutopia point
    #     - U  : utopia point
    #     - N  : number of samples for the approximation

    #    OUTPUT
    #     - hv : hypervolume

    n_sol, dim = F.shape
    samples = AU + ((U - AU) * np.random.rand(N, dim))
    valid = np.ones(N)
    dominated = 0
    for i in range(0, n_sol):
        idx = (np.sum((F[i, :] >= samples), 1) == dim) * valid
        dominated = dominated + np.sum(idx)
        valid[idx==True] = 0

    hv = dominated / N
    return hv