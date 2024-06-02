import numpy as np
from library.policies.Gaussian import Gaussian


class GaussianLinear(Gaussian):
    # GAUSSIANLINEAR Generic class for Gaussian policies. Its mean linearly
    # depends on some features, i.e., mu = A*[1; phi].

    def __init__(self):
        super().__init__()
        self.A = None
        self.U = None

    ## LOG(PDF)
    # def logpdf(self, Actions, States):
    #     ns = States.shape[1]
    #     d, na = Actions.shape
    #     assert(ns == na or na == 1 or ns == 1, 'Number of states and actions is not consistent.')
    #     phi = self.get_basis(States)
    #     mu = np.matmul(self.A * phi)
    #     Actions = Actions - mu
    #     Q = linalg.solve(self.U.T, Actions)
    #     q = np.sum(Q*Q, axis=0)
    #
    #     c = d * np.log(2 * np.pi) + logdet(self.Sigma, 'chol')
    #
    #     logprob = - (c + q) / 2
    #     return logprob

    ## MVNPDF
    # def evaluate(self, Actions, States):
    #     return np.exp(self.logpdf(Actions, States))

    ## MVNRND
    def drawAction(self, States):
        # Draw N samples, one for each state
        phi = self.get_basis(States)
        mu = np.matmul(self.A, phi)
        Actions = np.matmul(self.U.T, np.random.randn(self.daction, States.shape[1])) + mu
        return Actions
