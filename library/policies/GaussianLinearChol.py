import numpy as np
from scipy import linalg
from library.policies.GaussianLinear import GaussianLinear
from utils.rl.vec2mat import vec2mat


class GaussianLinearChol(GaussianLinear):
    # GAUSSIANLINEARCHOL Gaussian distribution with linear mean and constant
    # covariance: N(A*phi,S).
    # Parameters: mean A and Cholesky decomposition U, with S = U'U.
    #
    # U is stored row-wise, e.g:
    # U = [u1 u2 u3;
    #      0  u4 u5;
    #      0  0  u6] -> (u1 u2 u3 u4 u5 u6)

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
        self.basis = basis
        init_tri = initCholU.copy()
        init_tri = init_tri[np.tril(np.ones((dim, dim)), 0).T == 1]
        self.theta = np.concatenate((initA.T.ravel()[:, None], init_tri[:, None]), axis=0)
        self.dparams = len(self.theta)
        self.update(self.theta)

    ## Derivative of the logarithm of the policy
    def dlogPidtheta(self, state, action):
        # nsamples = state.shape[1]
        # phi = self.get_basis(state)
        # A = self.A
        # mu = np.matmul(A, phi)
        # cholU = self.U
        # invU = linalg.inv(cholU)
        # invUT = invU.T
        # invSigma = np.matmul(invU, invU.T)
        # diff = action - mu
        # dlogpdt_A = mtimescolumn(np.matmul(invSigma, diff), phi)
        # idx = np.tril(np.ones((self.daction, self.daction))).T
        # nelements = int(idx.sum())
        # tmp = -invUT.flatten()[:, None] + mtimescolumn(np.matmul(invUT, diff), np.matmul(invSigma, diff))
        # tmp = tmp.reshape(self.daction, self.daction, nsamples)
        # tmp = np.transpose(tmp, (2, 0, 1))
        # idx = np.tile(idx, (nsamples, 1, 1))
        # dlogpdt_cholU = tmp[idx == 1]
        # dlogpdt_cholU = dlogpdt_cholU.reshape((nsamples, nelements)).T
        # dlogpdt = np.concatenate((dlogpdt_A, dlogpdt_cholU), axis=0)
        # return dlogpdt
        return None

    ## WML
    def weightedMLUpdate(self, weights, Action, Phi):
        # assert (np.amin(weights) >= 0, 'Weights cannot be negative.')
        # assert (Phi.shape[0] == self.basis() + 1 * (not self.no_bias))
        # weights = weights / weights.sum(axis=1)
        # PhiW = Phi * weights
        # tmp = np.matmul(PhiW, Phi.T)
        # if np.rank(tmp) == Phi.shape[0]:
        #     A = np.matmul(linalg.solve(tmp, PhiW), Action.T)
        # else:
        #     A = np.matmul(np.matmul(linalg.pinv(tmp), PhiW), Action.T)
        #
        # A = A.T
        # diff = Action - np.matmul(A, Phi)
        # Sigma = np.sum(np.transpose(diff * weights, (0, 2, 1)) * np.transpose(diff, (2, 0, 1)), 2)
        # Z = (weights.sum(axis=1) ** 2 - (weights ** 2).sum(axis=1)) / weights.sum(axis=1)
        # Sigma = Sigma / Z
        # Sigma = nearestSPD(Sigma)
        # cholU = linalg.cholesky(Sigma)
        # tri = cholU.copy()
        # tri = tri[np.tril(np.ones((self.daction, self.daction)), 0).T == True].T
        # self.update(np.concatenate((A, tri[:, None]), axis=0))
        pass

    ## Update
    def update(self, arg, Sigma=None):
        if Sigma is None:
            theta = arg
            self.theta[:len(theta)] = theta
            n = len(self.theta) - np.sum(np.arange(1, self.daction + 1))
            A = vec2mat(self.theta[0:n], self.daction)
            indices = np.tril(np.ones((self.daction, self.daction))).T
            cholU = indices.copy()
            cholU[indices == 1] = self.theta[n:].flatten()
            self.A = A
            self.U = cholU
            self.Sigma = np.matmul(cholU.T, cholU)
        else:
            self.A = arg
            self.Sigma = Sigma
            U = linalg.cholesky(Sigma)
            self.U = U
            U = U.T
            U = U[np.tril(np.ones((self.daction, self.daction)), 0)].T
            self.theta = np.array([self.A, U.T])

    ## Change stochasticity
    def makeDeterministic(self):
        n = np.asarray(self.A).size
        self.theta[n:] = self.theta[n:] * 0
        self.update(self.theta)

    def randomize(self, factor=None):
        n = np.asarray(self.A).size
        self.theta[n:] = self.theta[n:] * factor
        self.update(self.theta)
