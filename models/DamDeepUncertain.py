import numpy as np
import random
from gym import spaces
from gym.utils import seeding


class DamDeepUncertain:

    def __init__(self, random_seed, ema):
        self.S = 1.0  # Reservoir surface
        self.W_IRR = 50.  # Water demand
        self.H_FLO_U = 50.  # Flooding threshold (upstream, i.e. height of dam)
        self.S_MIN_REL = 100.  # Release threshold (i.e. max capacity)
        self.DAM_INFLOW_MEAN = 40.  # Random inflow (e.g. rain)
        self.DAM_INFLOW_STD = 10.
        self.Q_MEF = 0.
        self.GAMMA_H2O = 1000.  # water density
        self.W_HYD = 4.36  # Hydroelectric demand
        self.Q_FLO_D = 30.  # Flooding threshold (downstream, i.e. releasing too much water)
        self.ETA = 1.  # Turbine efficiency
        self.G = 9.81  # Gravity

        self.dreward = 2
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,))
        self.action_space = spaces.Discrete(11)
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.dreward,))

        self.ema = ema
        self.random_seed = random_seed
        self.gamma = 1
        self.isAveraged = 1
        # self.utopia = np.array([0, -8.5])
        # self.antiutopia = np.array([-7, -12])
        self.seed(self.random_seed)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, n=None):
        # state = np.array([[self.s_init[i] for i in self.s_init_idx]])
        state = self.s_init
        return state

    def ema_reset(self, s_init):
        # self.s_init_idx = np.array([s_init_idx])
        self.s_init = np.array([[s_init]])
        return self.reset()

    def manifold_reset(self, scenarios):
        # self.s_init_idx = np.array(scenarios['s_init_idx'])
        self.s_init = np.array([scenarios['s_init']])
        return self.reset()

    def simulator(self, state, action, curr_step, last_action):
        if self.ema:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        else:
            action = action[0]
        # print(1/0)
        action *= 10
        nstates = state.shape[1]
        reward = np.zeros((self.dreward, nstates))

        # Bound the action
        actionLB = np.clip(state - self.S_MIN_REL, 0, None)
        actionUB = state

        # Penalty proportional to the violation
        bounded_action = np.clip(action, actionLB, actionUB)

        # Transition dynamic
        action = bounded_action
        dam_inflow = np.random.normal(self.DAM_INFLOW_MEAN, self.DAM_INFLOW_STD, nstates)
        nextstate = np.clip(state + dam_inflow - action, 0, None)

        # Cost due to the excess level w.r.t. a flooding threshold (upstream)
        reward[0, :] = -np.clip(nextstate / self.S - self.H_FLO_U, 0, None)
        # Deficit in the water supply w.r.t. the water demand
        reward[1, :] = -np.clip(self.W_IRR - action, 0, None)

        if curr_step == 100:
            absorb = np.ones(nstates)
        else:
            absorb = np.zeros(nstates)

        return nextstate, reward, absorb
