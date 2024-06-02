import numpy as np
from scipy import linalg
from library.policies.GaussianConstant import GaussianConstant
from utils.rl.nearestSPD import nearestSPD
from utils.rl.mtimescolumn import mtimescolumn


class GaussianConstantChol(GaussianConstant):
    # GAUSSIANCONSTANTCHOL Gaussian distribution with constant mean and
    # covariance: N(mu,S).
    # Parameters: mean mu and Cholesky decomposition U, with S = U'U.
    #
    # U is stored row-wise, e.g:
    # U = [u1 u2 u3;
    #      0  u4 u5;
    #      0  0  u6] -> (u1 u2 u3 u4 u5 u6)
    #
    # =========================================================================
    # REFERENCE
    # Y Sun, D Wierstra, T Schaul, J Schmidhuber
    # Efficient Natural Evolution Strategies (2009)

    ## Constructor

    def __init__(self, dim=None, initMean=None, initSigma=None):
        super().__init__()
        assert (np.isscalar(dim) and
                initMean.shape[0] == dim and
                initMean.shape[1] == 1 and
                initSigma.shape[0] == dim and
                initSigma.shape[1] == dim, 'Dimensions are not consistent.')

        initCholU = linalg.cholesky(initSigma)

        self.daction = dim
        init_tri = initCholU.copy()
        init_tri = init_tri[np.tril(np.ones((dim, dim)), 0).T == 1]
        self.theta = np.concatenate((initMean, init_tri[:, None]), axis=0)
        self.dparams = len(self.theta)
        self.update(self.theta)

    # Derivative of the logarithm of the policy
    def dlogPidtheta(self, action=None):
        nsamples = action.shape[1]
        mu = self.mu
        cholU = self.U
        invU = linalg.inv(cholU)
        invUT = invU.T
        invSigma = np.matmul(invU, invU.T)
        diff = action - mu
        dlogpdt_A = np.matmul(invSigma, diff)
        tmp = -invUT.flatten()[:, None] + mtimescolumn(np.matmul(invUT, diff), np.matmul(invSigma, diff))
        tmp = tmp.reshape(self.daction, self.daction, nsamples)
        tmp = np.transpose(tmp, (2, 0, 1))
        idx = np.tril(np.ones((self.daction, self.daction))).T
        nelements = int(idx.sum())
        idx = np.tile(idx, (nsamples, 1, 1))
        dlogpdt_cholU = tmp[idx == 1]
        dlogpdt_cholU = dlogpdt_cholU.reshape((nsamples, nelements)).T
        dlogpdt = np.concatenate((dlogpdt_A, dlogpdt_cholU), axis=0)
        return dlogpdt

    # WML
    def weightedMLUpdate(self, weights=None, Action=None):
        assert (np.amin(weights) >= 0, 'Weights cannot be negative.')
        weights = weights / weights.sum(axis=1)
        mu = np.matmul(Action, weights.T) / weights.sum(axis=1)
        diff = Action - mu
        Sigma = np.matmul(diff * weights, diff.T)
        Z = (weights.sum(axis=1) ** 2 - (weights ** 2).sum(axis=1)) / weights.sum(axis=1)
        Sigma = Sigma / Z
        Sigma = nearestSPD(Sigma)
        cholU = linalg.cholesky(Sigma)
        tri = cholU.copy()
        tri = tri[np.tril(np.ones((self.daction, self.daction)), 0).T == True].T
        self.update(np.concatenate((mu, tri[:, None]), axis=0))

    # FIM
    # def fisher(self):
    #     # Closed form Fisher information matrix
    #     cholU = self.U
    #     invU = np.linalg.inv(cholU)
    #     invSigma = invU * invU.T
    #     F_size = np.sum(range(1, self.daction + 1)) + self.daction
    #     F = np.zeros((F_size, F_size))
    #     F[0:self.daction, 0:self.daction] = invSigma
    #     idx = self.daction
    #     for k in range(self.daction):
    #         tmp = invSigma[k:, k:]
    #         tmp[0, 0] = tmp[0, 0] + 1 / cholU[k, k] ** 2
    #         step = tmp.shape[0]
    #         F[idx:idx + step - 1, idx:idx + step - 1] = tmp
    #         idx = idx + step
    #     return F

    def inverseFisher(self):
        # Closed form inverse Fisher information matrix
        cholU = self.U
        invU = linalg.inv(cholU)
        invSigma = np.matmul(invU, invU.T)
        invF_size = np.sum(range(1, self.daction + 1)) + self.daction
        invF = np.zeros((invF_size, invF_size))
        invF[0:self.daction, 0:self.daction] = self.Sigma
        idx = self.daction
        for k in range(self.daction):
            tmp = invSigma[k:, k:]
            tmp[0, 0] = tmp[0, 0] + 1 / cholU[k, k] ** 2
            step = tmp.shape[0]
            invF[idx:idx+step, idx:idx+step] = linalg.lstsq(tmp.T, np.eye(tmp.shape[0]).T)[0].T
            idx = idx + step
        invF[np.isnan(invF)] = 0
        return invF

    # Update
    def update(self, arg, Sigma=None):
        if Sigma is None:
            theta = arg
            self.theta[:len(theta)] = theta
            mu = self.theta[0: self.daction]
            indices = np.tril(np.ones((self.daction, self.daction))).T
            cholU = indices.copy()
            cholU[indices == 1] = self.theta[self.daction:].flatten()
            self.mu = mu
            self.Sigma = np.matmul(cholU.T, cholU)
            self.U = cholU
        else:
            self.mu = arg
            self.Sigma = Sigma
            U = linalg.cholesky(Sigma)
            self.U = U
            U = U.T
            U = U[np.tril(np.ones((self.daction, self.daction)), 0)].T
            self.theta = np.array([self.mu, U.T])

    # Change stochasticity
    def makeDeterministic(self):
        n = self.daction
        self.theta[n:] = self.theta[n:] * 0
        self.update(self.theta)

    def randomize(self, factor=None):
        n = self.daction
        self.theta[n:] = self.theta[n:] * factor
        self.update(self.theta)
