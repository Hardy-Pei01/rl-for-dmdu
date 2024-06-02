import gym
from gym import spaces
import numpy as np
import random
import math
from scipy.optimize import brentq
from gym.utils import seeding


class LakeDeepUncertain4:

    def __init__(self, random_seed, ema):
        self.alpha = 0.4
        self.steps = 100
        self.ema = ema
        self.random_seed = random_seed

        self.initial_state = np.array([0., 0., 0., 1.])

        self.action_space = spaces.Discrete(11)
        # self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
        #                                     high=np.array([15, 0.041, 1]), dtype=np.float)
        # self.reward_space = spaces.Box(low=np.array([0, 0]),
        #                                high=np.array([0.041, 0.11]), dtype=np.float)

        self.gamma = 1
        self.isAveraged = 0
        self.seed(self.random_seed)

    def get_natural_inflows(self, mean, stdev):
        return np.random.lognormal(
            math.log(mean ** 2 / math.sqrt(stdev ** 2 + mean ** 2)),
            math.sqrt(math.log(1.0 + stdev ** 2 / mean ** 2)),
            size=self.steps)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
        _, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, n):
        Pcrit = []
        natural_inflows = []
        for i in range(n):
            b = self.b[i]
            q = self.q[i]
            mean = self.mean[i]
            stdev = self.stdev[i]
            Pcrit.append(brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5))
            natural_inflows.append(self.get_natural_inflows(mean=mean, stdev=stdev))
        self.Pcrit = np.array(Pcrit)
        self.natural_inflows = np.array(natural_inflows).T
        state = np.tile(self.initial_state, (n, 1)).T
        return state

    def ema_reset(self, b, q, mean, stdev, delta):
        self.b = np.array([b])
        self.q = np.array([q])
        self.mean = np.array([mean])
        self.stdev = np.array([stdev])
        self.delta = np.array([delta])
        return self.reset(n=1)

    def manifold_reset(self, scenarios):
        self.b = np.array(scenarios['b'])
        self.q = np.array(scenarios['q'])
        self.mean = np.array(scenarios['mean'])
        self.stdev = np.array(scenarios['stdev'])
        self.delta = np.array(scenarios['delta'])
        return self.reset(n=len(self.b))

    def simulator(self, state, action, curr_step, last_action):
        if self.ema:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        else:
            action = action[0]
            last_action = last_action[0]
        action /= 100
        last_action /= 100
        nstates = state.shape[1]

        prev_p = state[0, :]
        next_P = \
            (1 - self.b) * prev_p + prev_p ** self.q / (1 + prev_p ** self.q) + \
            action + self.natural_inflows[curr_step - 1, :]
        reliability = (next_P < self.Pcrit)
        utility = self.alpha * action * np.power(self.delta, (curr_step - 1))
        if curr_step == 1:
            inertia = np.zeros(reliability.shape)
        else:
            inertia = abs(action - last_action) < 0.02
            if self.ema:
                inertia = np.array([inertia])

        nextstate = np.array([next_P, utility, inertia, reliability])
        reward = np.array([-next_P / self.steps, utility, inertia / (self.steps - 1), reliability / self.steps])

        if curr_step == self.steps:
            absorb = np.ones(nstates)
        elif curr_step < self.steps:
            absorb = np.zeros(nstates)
        else:
            raise RuntimeError

        return nextstate, reward, absorb
