import functools
import random
import numpy as np
import pandas as pd
from EMAWorkbench.em_framework import ScalarOutcome, MultiprocessingEvaluator
from EMAWorkbench.em_framework import sample_uncertainties
from EMAWorkbench.em_framework.optimization import EpsNSGAII, GenerationalBorg
from EMAWorkbench.util import ema_logging


def mean_variance(performance):
    mean = np.mean(performance)
    std = np.std(performance)
    mv = (mean + 1) / (std + 1)
    return mv


def ea_robust_training(nfe, random_seed, scenario_num, model, problem, scenario_path, restore_path,
                       borg, n_processes, results_path, checkpoint_path, r_metric, n_obj, dam=True):
    random.seed(random_seed)
    np.random.seed(random_seed)

    robust_model, convergence_metrics = model(problem, random_seed)

    fixed_scenarios_num = scenario_num
    scenarios_df_read = pd.read_csv(scenario_path)
    training_scenarios_designs = list(scenarios_df_read.to_records(index=False))
    training_scenarios = sample_uncertainties(robust_model, fixed_scenarios_num)
    training_scenarios.designs = training_scenarios_designs[:fixed_scenarios_num]

    percentile10 = functools.partial(np.percentile, q=10)
    MAXIMIZE = ScalarOutcome.MAXIMIZE
    if dam:
        obj_1 = "upstream_flooding"
        obj_2 = "water_demand"
    else:
        obj_1 = "utility"
        obj_2 = "reliability"
    if r_metric == "10th":
        robustnes_functions = [ScalarOutcome(f'10-p {obj_1}', kind=MAXIMIZE,
                                             variable_name=f'{obj_1}', function=percentile10),
                               ScalarOutcome(f'10-p {obj_2}', kind=MAXIMIZE,
                                             variable_name=f'{obj_2}', function=percentile10)]
    elif r_metric == "mv":
        robustnes_functions = [ScalarOutcome(f'mv {obj_1}', kind=MAXIMIZE,
                                             variable_name=f'{obj_1}', function=mean_variance),
                               ScalarOutcome(f'mv {obj_2}', kind=MAXIMIZE,
                                             variable_name=f'{obj_2}', function=mean_variance)]
    elif r_metric == "avg":
        if n_obj == 2:
            robustnes_functions = [ScalarOutcome(f'avg {obj_1}', kind=MAXIMIZE,
                                                 variable_name=f'{obj_1}', function=np.mean),
                                   ScalarOutcome(f'avg {obj_2}', kind=MAXIMIZE,
                                                 variable_name=f'{obj_2}', function=np.mean)]
        elif n_obj == 3:
            robustnes_functions = [ScalarOutcome(f'avg {obj_1}', kind=MAXIMIZE,
                                                 variable_name=f'{obj_1}', function=np.mean),
                                   ScalarOutcome(f'avg {obj_2}', kind=MAXIMIZE,
                                                 variable_name=f'{obj_2}', function=np.mean),
                                    ScalarOutcome('avg electricity_demand', kind=MAXIMIZE,
                                                  variable_name='electricity_demand', function=np.mean)]
    else:
        raise NotImplementedError

    ema_logging.log_to_stderr(ema_logging.INFO)

    if restore_path is not None:
        import pickle
        with open(restore_path, 'rb') as fh:
            restart_optimizer = pickle.load(fh)
        restart_optimizer.problem.function = lambda _: restart_optimizer.problem.function
        print("Restart")
    else:
        restart_optimizer = None
        print("Start")

    if borg:
        moea = GenerationalBorg
        print("Borg")
    else:
        moea = EpsNSGAII
        print("EpsNSGAII")

    with MultiprocessingEvaluator(robust_model, n_processes=n_processes) as evaluator:
        results, convergence, optimizer = evaluator.robust_optimize(
            algorithm=moea,
            robustness_functions=robustnes_functions,
            scenarios=training_scenarios,
            nfe=nfe,
            searchover='levers',
            epsilons=[0.01, 0.01],
            convergence=convergence_metrics,
            restart_optimizer=restart_optimizer
        )

    results.to_csv(f'{results_path}_policy.csv', index=False)
    convergence.to_csv(f'{results_path}_convergence.csv', index=False)

    # results = np.array(list(results.iloc[:, -2:].apply(tuple, axis=1)))
    print(results)
    print(convergence)

    # if checkpoint_path is not None:
    #     import pickle
    #
    #     optimizer.evaluator = None
    #     optimizer.algorithm.evaluator = None
    #     optimizer.problem.function = optimizer.problem.function(0)
    #     optimizer.problem.ema_constraints = None
    #
    #     with open(checkpoint_path, 'wb') as fh:
    #         pickle.dump(optimizer, fh)
