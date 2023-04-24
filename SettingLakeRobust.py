import numpy as np
import pandas as pd
from datetime import datetime
from models.MOMDP import MOMDP
from library.basis.lake_basis_poly import lake_basis_poly
from library.policies.Gibbs import Gibbs
from library.policies.GaussianConstantChol import GaussianConstantChol
from algs.RUN_Manifold import run, eval

for r_metric in ["10th", "avg"]:
    for random_seed in [3186775264, 3690172787, 462638671, 1926216712, 3087161096,
                        956500800, 2523200676, 1274569960, 2097286424, 3885705317,
                        562732020, 2861224539, 1350287007, 674137616, 3624030427,
                        703574460, 1883682950, 617160326, 3668976038, 96930842]:
    # for random_seed in [1926216712]:
        current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")
        scenarios = pd.read_csv("./results/lake_scenarios/train_scenarios_rl.csv")

        #  ======================== LOW LEVEL SETTINGS =========================  #
        problem = 'lake_deep_uncertain'
        if r_metric == "10th":
            utopia = np.array([2, 1])
            antiutopia = np.array([0, 0])
        elif r_metric == "mv":
            utopia = np.array([2, 2])
            antiutopia = np.array([1, 1])
        elif r_metric == "avg":
            utopia = np.array([2, 1])
            antiutopia = np.array([0, 0])
        else:
            raise NotImplementedError
        mdp = MOMDP(problem, seed=random_seed, utopia=utopia, antiutopia=antiutopia, dreward=2)
        robj = 1

        bfs = lake_basis_poly

        policy = Gibbs(bfs, np.zeros(((bfs() + 1) * mdp.actionUB, 1)), np.arange(mdp.actionLB, mdp.actionUB + 1))

        #  ======================= HIGH LEVEL SETTINGS =========================  #
        n_params = policy.dparams
        mu0 = policy.theta[0:n_params]
        Sigma0high = 10000 * np.eye(n_params)
        policy_high = GaussianConstantChol(n_params, mu0, Sigma0high)

        #  ======================== LEARNING SETTINGS ==========================  #
        episodes_eval = 1000
        steps_eval = 99
        episodes_learn = 200
        steps_learn = 99

        # N = 200
        # N_MAX = N * 5
        # MAXITER = 200
        # method = "repsep"
        # epsilon = 0.5

        N = 200
        N_MAX = N * 5
        MAXITER = 80
        method = "nes"
        epsilon = 0.3

        checkpoint_path = f"./results/rl_lake_robust_1/{method}_{r_metric}_{random_seed}_{N}_{MAXITER}_0{int(epsilon*10)}_{current_time}"
        start = datetime.now()
        best_policy, policy_high, hyperv_history = run(N, N_MAX, MAXITER, episodes_learn, steps_learn, policy, policy_high,
                                                       mdp, method, epsilon, checkpoint_path, scenarios, r_metric=r_metric)
        print(datetime.now() - start)

        # N_EVAL = 2000
        # evaluation_scenarios = pd.read_csv("./results/lake_scenarios/evaluation_scenarios.csv")
        # eval(N_EVAL, mdp, episodes_eval, steps_eval, policy, best_policy,
        #      policy_high, hyperv_history, evaluation_scenarios, r_metric=r_metric)
