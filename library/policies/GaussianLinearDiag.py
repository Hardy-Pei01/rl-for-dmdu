import numpy as np
from scipy import linalg
from library.policies.GaussianLinear import GaussianLinear
from utils.rl.vec2mat import vec2mat

class GaussianLinearDiag(GaussianLinear):
    # GAUSSIANLINEARDIAG Gaussian distribution with linear mean and constant
    # diagonal covariance: N(A*phi,S).
    # Parameters: mean A and diagonal std s, where S = diag(s)^2.

    ## Constructor
    def __init__(self, basis, dim, initA, initSigma, no_bias=None):
        super().__init__()
        if no_bias is None:
            no_bias = False

        self.no_bias = no_bias
        assert (np.isscalar(dim) and initA.shape[1] == basis() + 1 * (not no_bias),
                'Dimensions are not consistent.')

        initCholU = linalg.cholesky(initSigma)

        self.daction = dim
        initStd = np.diag(np.sqrt(initSigma))[:, None]
        self.basis = basis
        self.theta = np.concatenate((initA.T.ravel()[:, None], initStd), axis=0)
        self.dparams = len(self.theta)
        self.update(self.theta)
        
    ## Derivative of the logarithm of the policy
    def dlogPidtheta(self, state, action):
        return None

    ## Hessian of the logarithm of the policy
    def hlogPidtheta(self, state, action):
        return None
        
    ## WML
    def weightedMLUpdate(self, weights, Action, Phi):
        pass
        
    ## Update
    def update(self, arg, Sigma=None):
        if Sigma is None:
            theta = arg
            self.theta[:len(theta)] = theta
            n = len(self.theta) - self.daction
            A = vec2mat(self.theta[0:n], self.daction)
            std = self.theta[n:].ravel()
            self.A = A
            self.Sigma = np.diag(std ** 2)
            self.U = np.diag(std)
        else:
            assert np.isdiag(Sigma)
            self.A = arg
            std = np.sqrt(np.diag(Sigma))
            self.Sigma = np.diag(std ** 2)
            self.U = np.diag(std)
            self.theta = np.concatenate((self.A.T.ravel()[:, None], std), axis=0)
        
    ## Change stochasticity
    def makeDeterministic(self):
        self.theta[-self.daction:] = 0
        self.update(self.theta)
        
        
    def randomize(self, factor=None):
        self.theta[-self.daction:] = self.theta[-self.daction:] * factor
        self.update(self.theta)
        