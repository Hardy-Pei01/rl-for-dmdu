import numpy as np
from library.policies.PolicyDiscrete import PolicyDiscrete


class Gibbs(PolicyDiscrete):
    # GIBBS Gibbs (softmax) distribution with preferences on all but last
    # action. The temperature is fixed.

    # Constructor
    def __init__(self, basis=None, theta=None, action_list=None, no_bias=False):
        super().__init__()

        self.no_bias = no_bias
        assert ((basis() + 1 * (not no_bias)) * (len(action_list) - 1) == len(theta),
                'Wrong number of initial parameters.')

        self.basis = basis
        self.theta = theta
        self.action_list = action_list
        self.epsilon = 1
        self.daction = 1
        self.dparams = len(self.theta)

    # Distribution
    def distribution(self, States=None):
        Q = self.qFunction(States)
        Q = Q - np.amax(Q, 0)
        exp_term = np.exp(Q / self.epsilon)
        prob_list = exp_term * (1. / np.sum(exp_term, 0))
        prob_list = prob_list * (1. / np.sum(prob_list, 0))
        return prob_list

        ## Q-function

    def qFunction(self, States=None, Actions=None):
        # If no actions are provided, the function returns the Q-function
        # for all possible actions.
        nstates = States.shape[1]
        lactions = len(self.action_list)
        phi = self.get_basis(States)
        dphi = phi.shape[0]
        Q = np.concatenate((np.matmul(self.theta.reshape(lactions - 1, dphi), phi),
                            np.zeros((1, nstates))), axis=0)

        if States is not None and Actions is not None:
            idx = (np.arange(0, nstates - 1 + 1)) * lactions + Actions
            Q = Q[idx]

        return Q

    # Derivative of the logarithm of the policy
    # def dlogPidtheta(self, States=None, Actions=None):
    #     assert (States.shape[1] == Actions.shape[1], 'The number of states and actions must be the same.')
    #
    #     phi = self.get_basis(States)
    #     dphi = phi.shape[1 - 1]
    #     prob_list = self.distribution(States)
    #     dlpdt = -mtimescolumn(phi, prob_list[np.arange(0, len(prob_list) - 1), :]) / self.epsilon
    #     for i in np.arange(1, len(self.action_list) - 1 + 1).reshape(-1):
    #         idx1 = np.arange((i - 1) * dphi + 1, (i - 1) * dphi + dphi + 1)
    #         idx2 = Actions == i
    #         dlpdt[idx1, idx2] = dlpdt[idx1, idx2] + phi[:, idx2] / self.epsilon
    #
    #     return dlpdt

    # Phi(s,a) activates state-dependent features for a specific action
    # def get_basis_action(self, States=None, Actions=None):
    #     assert (States.shape[1] == Actions.shape[1], 'The number of states and actions must be the same.')
    #
    #     phi = self.get_basis(States)
    #     dphi, n = phi.shape
    #     phi_action = np.zeros((self.dparams, n))
    #     for i in np.arange(1, self.dparams / dphi + 1).reshape(-1):
    #         idx1 = np.arange((i - 1) * dphi + 1, (i - 1) * dphi + dphi + 1)
    #         idx2 = Actions == i
    #         phi_action[idx1, idx2] = phi_action[idx1, idx2] + phi[:, idx2] / self.epsilon
    #
    #     phi_action = phi_action / self.epsilon
    #     return phi_action

    # Update
    def update(self, theta=None):
        self.theta[:] = theta

    # Change stochasticity
    def makeDeterministic(self):
        self.epsilon = 1e-08

    def randomize(self, factor=None):
        self.theta = self.theta / factor
