import gym
from gym import spaces
import numpy as np
import random
import math
from scipy.optimize import brentq
from gym.utils import seeding


class LakeDiscrete1():

    def __init__(self, random_seed, ema):
        self.b = np.array([0.42])
        self.q = np.array([2])
        self.mean = np.array([0.02])
        self.stdev = np.array([0.0017])
        self.delta = np.array([0.98])
        self.alpha = 0.4

        self.steps = 99
        self.ema = ema
        self.random_seed = random_seed
        self.Pcrit = brentq(lambda x: x ** self.q / (1 + x ** self.q) - self.b * x, 0.01, 1.5)
        self.natural_inflows = self.get_natural_inflows()

        self.initial_state = np.array([0., 0., 1.])

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array([2.3, 0.041, 1]), dtype=np.float)
        self.reward_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([0.041, 0.11]), dtype=np.float)

        self.gamma = 1
        self.isAveraged = 0
        # self.utopia = np.array([1.8, 1])
        # self.antiutopia = np.array([0, 0])
        self.seed(self.random_seed)

    def get_natural_inflows(self):
        return np.full(self.steps, self.mean ** 2 / math.sqrt(self.stdev ** 2 + self.mean ** 2))

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, n):
        state = np.tile(self.initial_state, (n, 1)).T
        return state

    def simulator(self, state, action, curr_step, last_action):
        if self.ema:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        else:
            action = action[0]
        # print(1/0)
        action = action / 100
        nstates = state.shape[1]

        prev_p = state[0, :]
        next_P = \
            (1 - self.b) * prev_p + prev_p ** self.q / (1 + prev_p ** self.q) + \
            action + self.natural_inflows[curr_step - 1]
        reliability = (next_P < self.Pcrit)
        utility = self.alpha * action * np.power(self.delta, (curr_step - 1))
        # inertia = abs(action - last_action) > 0.02

        nextstate = np.array([next_P, utility, reliability])
        reward = np.array([utility, reliability / self.steps])

        if curr_step == self.steps:
            absorb = np.ones(nstates)
        else:
            absorb = np.zeros(nstates)

        return nextstate, reward, absorb
