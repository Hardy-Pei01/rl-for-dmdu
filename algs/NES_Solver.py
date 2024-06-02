import numpy as np
from scipy import linalg


class NES_Solver():
    # Natural Evolution Strategy with optimal baseline.
    # It supports Importance Sampling (IS).
    #
    # =========================================================================
    # REFERENCE
    # D Wierstra, T Schaul, T Glasmachers, Y Sun, J Peters, J Schmidhuber
    # Natural Evolution Strategy (2014)

    ## CLASS CONSTRUCTOR
    def __init__(self, epsilon=None):
        self.epsilon = epsilon
        return

    ## PERFORM AN OPTIMIZATION STEP
    def step(self, J, Actions, policy, W):
        nat_grad, stepsize = self.NESbase(policy, J, Actions, W)
        div = linalg.norm(nat_grad)
        policy.update(policy.theta + nat_grad * stepsize)
        return div

    ## CORE
    def NESbase(self, policy, J, Actions, W=None):
        dlogPidtheta = policy.dlogPidtheta(Actions)
        # J = np.transpose(J, (2, 1, 0))
        den = np.sum(dlogPidtheta**2 * W**2, 1)
        num = np.sum(dlogPidtheta**2 * (J * W**2), 1)
        b = num * (1/den)
        b[np.isnan(b)] = 0
        b = b[:, None]
        diff = (J - b) * W
        grad = np.sum(dlogPidtheta * diff, 1)
        # N = np.sum(W); # lower variance
        N = len(W)

        grad = grad / N

        # If we can compute the FIM in closed form, we use it
        # F = policy.fisher()

        # If we can compute the FIM inverse in closed form, we use it
        invF = policy.inverseFisher()
        nat_grad = np.matmul(invF, grad)[:, None]

        lambda_ = np.sqrt(np.diag(np.matmul(grad, nat_grad)) / (8 * self.epsilon)).T
        lambda_ = np.maximum(lambda_, 1e-08)

        stepsize = 1 / (2 * lambda_)
        return nat_grad, stepsize
