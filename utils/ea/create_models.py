from EMAWorkbench.em_framework import (Model, RealParameter, ScalarOutcome, ArrayOutcome,
                                       IntegerParameter, Constant)
from EMAWorkbench.em_framework.optimization import (EpsilonProgress)
from functools import partial
# from models.Lake2 import Lake1
from models.LakeDiscrete import LakeDiscrete1
from models.LakeDiscrete4 import LakeDiscrete4
# from models.LakeUncertain import LakeUncertain1
from models.LakeDeepUncertain import LakeDeepUncertain
from models.LakeDeepUncertain4 import LakeDeepUncertain4
from models.LakeDeepUncertain4_test import LakeDeepUncertain4_test
# from models.Dam2 import Dam1
from models.DamDiscrete import DamDiscrete1
from models.Dam3Discrete import Dam3Discrete
from models.DamUncertain import DamUncertain1
from models.DamDeepUncertain import DamDeepUncertain
from models.Dam3Deep import Dam3DeepUncertain
from models.Fishing import Fishing
from models.Fishing3 import Fishing3


# def create_lake_training(problem, random_seed):
#     # instantiate the model
#     env = Lake1(random_seed, ema=True)
#     problem = partial(problem, env)
#     theModel = Model('lakeproblem', function=problem)
#     theModel.time_horizon = 100
#
#     theModel.outcomes = [ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
#                          ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]
#
#     # set levers, one for each time step
#     theModel.levers = [RealParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]
#     # theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]
#
#     convergence_metrics = [EpsilonProgress()]
#
#     theModel.constantcs = [Constant('alpha', 0.4)]
#
#     return theModel, convergence_metrics


def create_lake_discrete_training(problem, random_seed):
    # instantiate the model
    env = LakeDiscrete1(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('lakeDiscreteProblem', function=problem)
    theModel.time_horizon = 100

    theModel.outcomes = [ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    theModel.constantcs = [Constant('alpha', 0.4)]

    return theModel, convergence_metrics


def create_lake4_discrete_training(problem, random_seed):
    # instantiate the model
    env = LakeDiscrete4(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('lakeDiscreteProblem', function=problem)
    theModel.time_horizon = 100

    theModel.outcomes = [ScalarOutcome('avg_pollution',
                                       kind=ScalarOutcome.MINIMIZE),
                         ScalarOutcome('utility',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('inertia',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('reliability',
                                       kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    theModel.constantcs = [Constant('alpha', 0.4)]

    return theModel, convergence_metrics


# def create_lake_robust_training(problem, random_seed):
#     # instantiate the model
#     env = LakeUncertain1(random_seed, ema=True)
#     problem = partial(problem, env)
#     theModel = Model('lakeRobustProblem', function=problem)
#     theModel.time_horizon = 100
#
#     # specify uncertainties
#     # theModel.uncertainties = [RealParameter('b', 0.1, 0.45)]
#     theModel.uncertainties = [RealParameter('mean', 0.01, 0.025),
#                               RealParameter('stdev', 0.001, 0.002),
#                               RealParameter('b', 0.35, 0.45),
#                               RealParameter('q', 2, 2.8),
#                               RealParameter('delta', 0.97, 0.99)]
#
#     theModel.outcomes = [ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
#                          ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]
#
#     # set levers, one for each time step
#     theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]
#
#     convergence_metrics = [EpsilonProgress()]
#
#     theModel.constantcs = [Constant('alpha', 0.4)]
#
#     return theModel, convergence_metrics


def create_lake_deep_training(problem, random_seed):
    # instantiate the model
    env = LakeDeepUncertain(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('lakeDeepProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    theModel.uncertainties = [RealParameter('b', 0.1, 0.45),
                              RealParameter('q', 2.0, 4.5),
                              RealParameter('mean', 0.01, 0.05),
                              RealParameter('stdev', 0.001, 0.005),
                              RealParameter('delta', 0.93, 0.99)]

    theModel.outcomes = [ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    theModel.constantcs = [Constant('alpha', 0.4)]

    return theModel, convergence_metrics


def create_lake4_deep_training(problem, random_seed):
    # instantiate the model
    env = LakeDeepUncertain4(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('lakeDeepProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    theModel.uncertainties = [RealParameter('b', 0.1, 0.45),
                              RealParameter('q', 2.0, 4.5),
                              RealParameter('mean', 0.01, 0.05),
                              RealParameter('stdev', 0.001, 0.005),
                              RealParameter('delta', 0.93, 0.99)]

    theModel.outcomes = [ScalarOutcome('avg_pollution',
                                       kind=ScalarOutcome.MINIMIZE),
                         ScalarOutcome('utility',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('inertia',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('reliability',
                                       kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    theModel.constantcs = [Constant('alpha', 0.4)]

    return theModel, convergence_metrics


def create_lake4_deep_training_test(problem, random_seed):
    # instantiate the model
    env = LakeDeepUncertain4_test(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('lakeDeepProblem', function=problem)
    theModel.time_horizon = 10

    # specify uncertainties
    theModel.uncertainties = [RealParameter('b', 0.1, 0.45),
                              RealParameter('q', 2.0, 4.5),
                              RealParameter('mean', 0.01, 0.05),
                              RealParameter('stdev', 0.001, 0.005),
                              RealParameter('delta', 0.93, 0.99)]

    theModel.outcomes = [ScalarOutcome('avg_pollution',
                                       kind=ScalarOutcome.MINIMIZE),
                         ScalarOutcome('utility',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('inertia',
                                       kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('reliability',
                                       kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    theModel.constantcs = [Constant('alpha', 0.4)]

    return theModel, convergence_metrics


# def create_dam_training(problem, random_seed):
#     # instantiate the model
#     env = Dam1(random_seed, ema=True)
#     problem = partial(problem, env)
#     theModel = Model('damproblem', function=problem)
#     theModel.time_horizon = 100
#
#     theModel.outcomes = [ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
#                          ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]
#
#     # set levers, one for each time step
#     theModel.levers = [RealParameter(str(i), 0, 140) for i in range(theModel.time_horizon)]
#
#     convergence_metrics = [EpsilonProgress()]
#
#     return theModel, convergence_metrics


def create_dam_discrete_training(problem, random_seed):
    # instantiate the model
    env = DamDiscrete1(random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=2)
    theModel = Model('damDiscreteProblem', function=problem)
    theModel.time_horizon = 100

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam3_discrete_training(problem, random_seed):
    # instantiate the model
    env = Dam3Discrete(dreward=3, random_seed=random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=3)
    theModel = Model('dam3DiscreteProblem', function=problem)
    theModel.time_horizon = 100

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('electricity_demand', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam4_discrete_training(problem, random_seed):
    # instantiate the model
    env = Dam3Discrete(dreward=4, random_seed=random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=4)
    theModel = Model('dam4DiscreteProblem', function=problem)
    theModel.time_horizon = 100

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('electricity_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('downstream_flooding', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam_robust_training(problem, random_seed):
    # instantiate the model
    env = DamUncertain1(random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=2)
    theModel = Model('damRobustProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    # theModel.uncertainties = [IntegerParameter('s_init_idx', 0, 9)]
    theModel.uncertainties = [RealParameter('s_init', 120, 200)]

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam_deep_training(problem, random_seed):
    # instantiate the model
    env = DamDeepUncertain(random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=2)
    theModel = Model('damDeepProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    # theModel.uncertainties = [IntegerParameter('s_init_idx', 0, 9)]
    theModel.uncertainties = [RealParameter('s_init', 120, 200)]

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam3_deep_training(problem, random_seed):
    # instantiate the model
    env = Dam3DeepUncertain(random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=3)
    theModel = Model('dam3DeepProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    theModel.uncertainties = [RealParameter('s_init', 120, 200)]

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('electricity_demand', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_dam4_deep_training(problem, random_seed):
    # instantiate the model
    env = Dam3DeepUncertain(dreward=4, random_seed=random_seed, ema=True)
    problem = partial(problem, env=env, n_obj=4)
    theModel = Model('dam4DeepProblem', function=problem)
    theModel.time_horizon = 100

    # specify uncertainties
    theModel.uncertainties = [RealParameter('s_init', 120, 200)]

    theModel.outcomes = [ScalarOutcome('upstream_flooding', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('water_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('electricity_demand', kind=ScalarOutcome.MAXIMIZE),
                         ScalarOutcome('downstream_flooding', kind=ScalarOutcome.MAXIMIZE)]

    # set levers, one for each time step
    theModel.levers = [IntegerParameter(str(i), 0, 10) for i in range(theModel.time_horizon)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_fishing_training(problem, random_seed):
    # instantiate the model
    env = Fishing(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('fishingProblem', function=problem)
    theModel.time_horizon = 100
    daction = env.daction

    theModel.outcomes = [ScalarOutcome(f'fish_population_{i+1}', kind=ScalarOutcome.MAXIMIZE) for i in range(daction)]

    # set levers, one for each time step
    theModel.levers = [RealParameter(str(i), 0, 1) for i in range(theModel.time_horizon * daction)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics


def create_fishing3_training(problem, random_seed):
    # instantiate the model
    env = Fishing3(random_seed, ema=True)
    problem = partial(problem, env)
    theModel = Model('fishingProblem', function=problem)
    theModel.time_horizon = 100
    daction = env.daction

    theModel.outcomes = [ScalarOutcome(f'fish_population_{i+1}', kind=ScalarOutcome.MAXIMIZE) for i in range(daction)]

    # set levers, one for each time step
    theModel.levers = [RealParameter(str(i), 0, 1) for i in range(theModel.time_horizon * daction)]

    convergence_metrics = [EpsilonProgress()]

    return theModel, convergence_metrics
