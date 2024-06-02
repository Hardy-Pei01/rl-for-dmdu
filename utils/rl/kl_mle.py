import numpy as np


def kl_mle(pWeighting, qWeighting):
    # KL_MLE Approximates the Kullback-Leibler KL(Q||P) divergence beetween two
    # distributions Q (the old one) and P (the new one) using the weights for
    # a Maximum-Likelihood update.
    # If no weights for Q are provided, they are assumed to be 1.

    qWeighting = qWeighting / qWeighting.sum()
    pWeighting = pWeighting / pWeighting.sum()
    index = pWeighting > (10 ** (-10))
    qWeighting = qWeighting[index]
    pWeighting = pWeighting[index]
    div = np.sum(pWeighting * np.log(pWeighting / qWeighting))
    return div
