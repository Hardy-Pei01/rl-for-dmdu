import numpy as np
from datetime import datetime
from models.MOMDP import MOMDP
from library.basis.dam_basis_rbf import dam_basis_rbf
from library.policies.Gibbs import Gibbs
from library.policies.GaussianConstantChol import GaussianConstantChol
from algs.RUN_Manifold import run, eval

random_seed = 1
current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")

#  ======================== LOW LEVEL SETTINGS =========================  #
problem = 'dam3_discrete'
r_metric = "avg"
utopia = np.array([0, -9, 0])
antiutopia = np.array([-1, -10, -1])
mdp = MOMDP(problem, seed=random_seed, utopia=utopia, antiutopia=antiutopia)
robj = 1

bfs = dam_basis_rbf

policy = Gibbs(bfs, np.zeros(((bfs() + 1) * mdp.actionUB, 1)), np.arange(mdp.actionLB, mdp.actionUB + 1))

#  ======================= HIGH LEVEL SETTINGS =========================  #
n_params = policy.dparams
mu0 = policy.theta[0:n_params]
Sigma0high = 2500 * np.eye(n_params)
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high)

#  ======================== LEARNING SETTINGS ==========================  #
episodes_eval = 1
steps_eval = 100
episodes_learn = 1
steps_learn = 100

N = 100
N_MAX = N * 5
MAXITER = 16
method = "repsep"
epsilon = 0.5

# N = 100
# N_MAX = N * 5
# MAXITER = 16
# method = "nes"
# epsilon = 0.2

checkpoint_path = f"./results/rl_dam/repsep_{N}_{MAXITER}_0{int(epsilon*10)}_{current_time}"
start = datetime.now()
best_policy, policy_high, hyperv_history = run(N, N_MAX, MAXITER, episodes_learn, steps_learn, policy, policy_high,
                                               mdp, method, epsilon, checkpoint_path, None, r_metric=r_metric)
print(datetime.now() - start)

N_EVAL = 1000
eval(N_EVAL, mdp, episodes_eval, steps_eval, policy, best_policy,
     policy_high, hyperv_history, None, r_metric=r_metric)
