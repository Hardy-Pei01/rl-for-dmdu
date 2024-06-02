# from models.Dam2 import Dam1
from models.DamDiscrete import DamDiscrete1
from models.Dam3Discrete import Dam3Discrete
from models.DamUncertain import DamUncertain1
from models.DamDeepUncertain import DamDeepUncertain
from models.Dam3Deep import Dam3DeepUncertain
# from models.Lake2 import Lake1
from models.LakeDiscrete import LakeDiscrete1
from models.LakeDiscrete4 import LakeDiscrete4
from models.LakeDeepUncertain import LakeDeepUncertain
from models.LakeDeepUncertain4 import LakeDeepUncertain4
from models.Fishing import Fishing
from models.Fishing3 import Fishing3


class MOMDP():
    def __init__(self, env, seed, dstate, dreward, daction=1, utopia=None, antiutopia=None):
        # if env == "dam":
        #     self.env = Dam1(random_seed=seed, ema=False)
        if env == "dam_discrete":
            self.env = DamDiscrete1(random_seed=seed, ema=False)
        elif env == "dam3_discrete":
            self.env = Dam3Discrete(dreward=dreward, random_seed=seed, ema=False)
        elif env == "dam_uncertain":
            self.env = DamUncertain1(random_seed=seed, ema=False)
        elif env == "dam_deep_uncertain":
            self.env = DamDeepUncertain(random_seed=seed, ema=False)
        elif env == "dam3_deep":
            self.env = Dam3DeepUncertain(dreward=dreward, random_seed=seed, ema=False)
        elif env == "lake_discrete":
            self.env = LakeDiscrete1(random_seed=seed, ema=False)
        elif env == "lake4_discrete":
            self.env = LakeDiscrete4(random_seed=seed, ema=False)
        elif env == "lake_deep_uncertain":
            self.env = LakeDeepUncertain(random_seed=seed, ema=False)
        elif env == "lake4_deep_uncertain":
            self.env = LakeDeepUncertain4(random_seed=seed, ema=False)
        elif env == "fishing":
            self.env = Fishing(random_seed=seed, ema=False)
        elif env == "fishing3":
            self.env = Fishing3(random_seed=seed, ema=False)
        else:
            raise NotImplementedError
        if env == "fishing" or env == "fishing3":
            self.actionUB = None
            self.actionLB = None
            # self.stateUB = self.env.observation_space.high
            # self.stateLB = self.env.observation_space.low
        else:
            self.actionUB = self.env.action_space.n - 1
            self.actionLB = 0
            # self.stateUB = None
            # self.stateLB = None
        # self.stateUB = self.env.observation_space.high[:, None]
        # self.stateLB = self.env.observation_space.low[:, None]
        self.dstate = dstate
        self.daction = daction
        self.dreward = dreward
        self.utopia = utopia
        self.antiutopia = antiutopia
        self.isAveraged = self.env.isAveraged
        self.gamma = self.env.gamma

    def reset(self, n):
        return self.env.reset(n)

    def manifold_reset(self, scenarios):
        return self.env.manifold_reset(scenarios)

    def simulator(self, state, action, curr_step, last_action):
        return self.env.simulator(state, action, curr_step, last_action)
