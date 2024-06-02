import numpy as np
from scipy import linalg
from library.policies.Gaussian import Gaussian


class GaussianConstant(Gaussian):
    # GAUSSIANCONSTANT Generic class for Gaussian policies. Its parameters do
    # not depend on any features, i.e., Action = N(mu, Sigma).

    def __init__(self):
        super().__init__()
        self.mu = None
        self.U = None

    # LOG(PDF)
    def logpdf(self, Actions=None):
        Actions = Actions - self.mu
        d = Actions.shape[0]
        Q = linalg.solve(self.U.T, Actions)
        q = np.sum(Q * Q, axis=0)

        c = d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(self.U)))

        logprob = - (c + q) / 2
        return logprob

    # MVNPDF
    def evaluate(self, Actions=None):
        return np.exp(self.logpdf(Actions))

    # MVNRND
    def drawAction(self, N):
        Actions = np.matmul(self.U.T, np.random.randn(self.daction, N)) + self.mu
        return Actions
