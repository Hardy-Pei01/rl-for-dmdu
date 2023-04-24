import random
import numpy as np
from EMAWorkbench.em_framework import MultiprocessingEvaluator
from EMAWorkbench.em_framework.optimization import EpsNSGAII, GenerationalBorg
from EMAWorkbench.util import ema_logging


def ea_training(nfe, random_seed, model, problem, borg, n_processes,
                results_path, restore_path, checkpoint_path):
    random.seed(random_seed)
    np.random.seed(random_seed)

    training_model, convergence_metrics = model(problem, random_seed)

    # ema_logging.log_to_stderr(ema_logging.INFO)

    if restore_path is not None:
        import pickle
        with open(restore_path, 'rb') as fh:
            restart_optimizer = pickle.load(fh)
        restart_optimizer.problem.function = lambda _:restart_optimizer.problem.function
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

    with MultiprocessingEvaluator(training_model, n_processes=n_processes) as evaluator:
        results, convergence, optimizer = evaluator.optimize(
            algorithm=moea,
            nfe=nfe,
            searchover='levers',
            epsilons=[0.01, 0.01],
            convergence=convergence_metrics,
            restart_optimizer=restart_optimizer
        )

    results.to_csv(f'{results_path}_policy.csv', index=False)
    convergence.to_csv(f'{results_path}_convergence.csv', index=False)

    print(results)
    print(convergence)

    # import pickle
    #
    # optimizer.evaluator = None
    # optimizer.algorithm.evaluator = None
    # optimizer.problem.function = optimizer.problem.function(0)
    # optimizer.problem.ema_constraints = None
    #
    # with open(checkpoint_path, 'wb') as fh:
    #     pickle.dump(optimizer, fh)