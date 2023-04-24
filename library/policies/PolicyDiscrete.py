import math
import numpy as np
from library.policies.Policy import Policy
from utils.rl.mymnrnd import mymnrnd


class PolicyDiscrete(Policy):

    def __init__(self):
        super().__init__()
        self.action_list = None

    def distribution(self, States=None):
        pass

    def qFunction(self, States=None):
        pass

    def vFunction(self, States=None):
        prob_list = self.distribution(States)
        Q = self.qFunction(States)
        V = np.sum(np.multiply(Q, prob_list))
        return V

    # def evaluate(self, States=None, Actions=None):
    #     # Evaluate pairs (state, action)
    #     assert (len(Actions) == States.shape[1], 'The number of states and actions must be the same.')
    #     idx = np.array([self.action_list.index(action) for action in Actions])
    #
    #     # Get action probability
    #     prob_list = self.distribution(States)
    #     nlist = len(self.action_list)
    #     naction = len(Actions)
    #     idx = (np.arange(1, naction * nlist + nlist, nlist)) + idx - 1
    #     prob_list = prob_list
    #     probability = np.transpose(prob_list(idx))
    #     return probability

    def drawAction(self, States=None):
        prob_list = self.distribution(States)
        Actions = mymnrnd(prob_list, States.shape[1])

        return Actions

    def entropy(self, States=None):
        prob_list = self.distribution(States)
        idx = np.logical_or(np.logical_or(np.isinf(prob_list), np.isnan(prob_list)), prob_list) == 0
        prob_list[idx] = 1

        S = - np.sum(np.multiply(prob_list, np.array([math.log(each, 2) for each in prob_list])), 0)
        S = np.mean(S)
        return S
