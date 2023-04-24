import math
import numpy as np
from library.policies.Policy import Policy
from utils.rl.logdet import logdet


class Gaussian(Policy):
    # GAUSSIAN Generic class for Gaussian distributions.

    def __init__(self):
        super().__init__()
        self.daction = None
        self.Sigma = None

    def entropy(self, varargin=None):
        # Differential entropy, can be negative
        S = 0.5 * (self.daction + self.daction * np.log(2 * math.pi) + logdet(self.Sigma, 'chol'))
        return S
