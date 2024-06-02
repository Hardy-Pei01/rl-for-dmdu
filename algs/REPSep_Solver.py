import numpy as np
from scipy.optimize import least_squares
from utils.rl.kl_mle import kl_mle


class REPSep_Solver():

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.eta = 1000.0

    # PERFORM AN OPTIMIZATION STEP
    def step(self, J, Actions, policy, W):
        d, divKL = self.optimize(J, W)
        policy.weightedMLUpdate(d, Actions)
        return divKL

    # CORE
    def optimize(self, J, W):
        # Optimization problem settings
        self.eta = least_squares(self.dual, args=(J[0], W), x0=self.eta, jac='cs', bounds=(1e-8, 1e8),
                                 method='trf', ftol=1e-12, xtol=1e-8, max_nfev=5000)['x'][0]

        # Compute weights for weighted ML update
        d = W * np.exp((J - np.amax(J)) / self.eta)

        # Compute KL divergence
        qWeighting = W
        pWeighting = d[0]
        divKL = kl_mle(pWeighting, qWeighting)
        return d, divKL

    # DUAL FUNCTION
    def dual(self, eta, J, W):
        n = np.sum(W)
        maxJ = np.amax(J)
        weights = W * np.exp((J - maxJ) / eta)

        sumWeights = np.sum(weights)
        g = eta * self.epsilon + eta * np.log(sumWeights / n) + maxJ

        return g
