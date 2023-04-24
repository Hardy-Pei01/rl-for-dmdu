import numpy as np
import random
from gym import spaces
from gym.utils import seeding


class Fishing3:
    metadata = {"render.modes": ["human"]}

    def __init__(self, random_seed, ema):

        # parameters
        self.n_species = 3
        self.r = np.array([0.25, 0.3, 0.35])
        self.alpha = np.array([[1, 0.1, 0.2], [0.2, 1, 0.1], [0.2, 0.2, 1]])
        self.init_state = np.array([0.5, 0.5, 0.5])

        # Preserve these for reset
        self.Tmax = 100

        self.daction = self.n_species
        self.dreward = self.n_species
        self.action_space = spaces.Box(
            np.array([0] * self.daction, dtype=np.float32),
            np.array([1] * self.daction, dtype=np.float32),
            dtype=np.float32,
            shape=(self.daction,)
        )
        self.observation_space = spaces.Box(
            np.array([0] * self.n_species, dtype=np.float32),
            np.array([1] * self.n_species, dtype=np.float32),
            dtype=np.float32,
            shape=(self.n_species,)
        )
        self.reward_space = spaces.Box(
            np.array([0] * self.dreward, dtype=np.float32),
            np.array([1] * self.dreward, dtype=np.float32),
            dtype=np.float32,
            shape=(self.dreward,)
        )

        self.ema = ema
        self.random_seed = random_seed
        self.gamma = 1
        self.isAveraged = 0
        self.seed(self.random_seed)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def simulator(self, state, action, curr_step, last_action):
        nstates = state.shape[1]
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        # Apply harvest and population growth
        harvest, harvested_fish_population = self.harvest_draw(state, action)
        next_fish_population = self.population_draw(harvested_fish_population, nstates)

        # should be the instanteous reward, not discounted
        reward = harvest
        if curr_step >= self.Tmax:
            done = np.ones(nstates)
        else:
            done = np.zeros(nstates)

        return next_fish_population, reward, done

    def reset(self, n):
        fish_population = np.tile(self.init_state, (n, 1)).T

        return fish_population

    def ema_reset(self, s_init):
        return None

    def manifold_reset(self, scenarios):
        return None

    def harvest_draw(self, fish_population, quota):
        """
        Select a value to harvest at each time step.
        """

        harvest = np.minimum(fish_population, quota)
        harvested_fish_population = np.clip(fish_population - harvest, 0.0, None)
        return harvest, harvested_fish_population

    def population_draw(self, harvested_fish_population, nstates):
        """
        Select a value for population to grow or decrease at each time step.
        """

        next_fish_population = np.zeros((self.n_species, nstates))
        for i in range(self.n_species):
            interference = np.dot(self.alpha[i, :], harvested_fish_population)
            next_fish_population[i, :] = harvested_fish_population[i]\
                                         + self.r[i] * harvested_fish_population[i]\
                                         * (1.0 - interference)
        return next_fish_population
