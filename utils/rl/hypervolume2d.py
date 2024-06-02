import numpy as np
from utils.rl.pareto import pareto
from utils.rl.normalize_data import normalize_data


def hypervolume2d(f=None, antiutopia=None, utopia=None):
    # HYPERVOLUME2D Computes the hypervolume of a 2-dimensional frontier for a
    # maximization problem. If the user provides both the utopia and antiutopia,
    # the frontier is normalized in order to have the objectives in the range
    # [0, 1]. If only the antiutopia point is provided, the frontier is not
    # normalized and the antiutopia is chosen as reference point.
    # Please note that points at the same level or below the antiutopia are not
    # considered, so choose the antiutopia carefully. For example, if the
    # antiutopia is [0,-19], the point [124, -19] is ignored, so it would be
    # better to choose [124, -20] as antiutopia.
    #
    #    INPUT
    #     - f          : [N x D] matrix representing a D-dimensional Pareto
    #                    front of N points
    #     - antiutopia : [1 x D] vector of antiutopia point
    #     - utopia     : [1 x D] vector of utopia point
    #
    #    OUTPUT
    #     - hv         : hypervolume

    f, _, _ = pareto(f)

    if utopia is not None:
        f = normalize_data(f, antiutopia, utopia)
        r = np.zeros((1, f.shape[1]))
    else:
        r = antiutopia

    # If a solution lays below the reference point, ignore it
    isBelow = np.sum((r > f), 1) >= 1
    f = np.delete(f, isBelow==True, axis=0)
    if len(f) == 0:
        return 0

    f = f[f[:, 0].argsort(),]
    b = np.diff(np.concatenate((r[:, 0], f[:, 0]), axis=0))
    h = f[:, 1] - r[:, 1]
    hv = sum(np.multiply(b, h))
    return hv
