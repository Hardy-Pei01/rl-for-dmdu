import numpy as np
from datetime import datetime
from models.MOMDP import MOMDP
from library.basis.lake_basis_poly import lake_basis_poly
from library.basis.basis_poly import basis_poly
from library.policies.Gibbs import Gibbs
from library.policies.GaussianConstantChol import GaussianConstantChol
from algs.RUN_Manifold import run, eval

for random_seed in [3186775264, 3690172787, 462638671, 1926216712, 3087161096,
                    956500800, 2523200676, 1274569960, 2097286424, 3885705317,
                    562732020, 2861224539, 1350287007, 674137616, 3624030427,
                    703574460, 1883682950, 617160326, 3668976038, 96930842]:
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")

    #  ======================== LOW LEVEL SETTINGS =========================  #
    problem = 'lake_discrete'
    r_metric = "avg"
    utopia = np.array([1.8, 1])
    antiutopia = np.array([0, 0])
    mdp = MOMDP(problem, seed=random_seed, utopia=utopia, antiutopia=antiutopia, dreward=2)
    robj = 1

    bfs = lake_basis_poly
    # bfs = lambda state=None: basis_poly(1, mdp.dstate, 0, state)

    policy = Gibbs(bfs, np.zeros(((bfs() + 1) * mdp.actionUB, 1)), np.arange(mdp.actionLB, mdp.actionUB + 1))

    #  ======================= HIGH LEVEL SETTINGS =========================  #
    n_params = policy.dparams
    mu0 = policy.theta[0:n_params]
    Sigma0high = 10000 * np.eye(n_params)
    policy_high = GaussianConstantChol(n_params, mu0, Sigma0high)

    #  ======================== LEARNING SETTINGS ==========================  #
    episodes_eval = 1
    steps_eval = 99
    episodes_learn = 1
    steps_learn = 99

    N = 400
    N_MAX = N * 5
    MAXITER = 200
    method = "repsep"
    epsilon = 0.5

    # N = 200
    # N_MAX = N * 5
    # MAXITER = 200
    # if random_seed == 3690172787 or random_seed == 3624030427 or random_seed == 3885705317:
    #     MAXITER = 160
    # if random_seed == 562732020 or random_seed == 2861224539 or random_seed == 674137616:
    #     MAXITER = 140
    # method = "nes"
    # epsilon = 0.3

    checkpoint_path = f"./results/rl_lake_1/{method}_{random_seed}_{N}_{MAXITER}_0{int(epsilon*10)}_{current_time}"
    start = datetime.now()
    best_policy, policy_high, hyperv_history = run(N, N_MAX, MAXITER, episodes_learn, steps_learn, policy, policy_high,
                                                   mdp, method, epsilon, checkpoint_path, None, r_metric=r_metric)
    print(datetime.now() - start)

    # N_EVAL = 2000
    # eval(N_EVAL, mdp, episodes_eval, steps_eval, policy, best_policy, policy_high, hyperv_history, None, r_metric=r_metric)