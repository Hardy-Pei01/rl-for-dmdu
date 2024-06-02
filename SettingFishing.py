import numpy as np
from datetime import datetime
from models.MOMDP import MOMDP
from library.basis.basis_poly import basis_poly
from library.basis.fishing_basis_poly import fishing_basis_poly
from utils.rl.nearestSPD import nearestSPD
from library.policies.GaussianLinearChol import GaussianLinearChol
from library.policies.GaussianLinearDiag import GaussianLinearDiag
from library.policies.GaussianConstantChol import GaussianConstantChol
from algs.RUN_Manifold import run, eval

random_seed = 1
current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")

#  ======================== LOW LEVEL SETTINGS =========================  #
problem = 'fishing'
r_metric = "avg"
utopia = np.array([11, 11])
antiutopia = np.array([0, 0])
mdp = MOMDP(problem, seed=random_seed, daction=2, utopia=utopia, antiutopia=antiutopia)

# bfs = lambda state=None: basis_poly(1, mdp.dstate, 0, state)
bfs = fishing_basis_poly

A0 = np.zeros((mdp.daction, bfs()+1))
Sigma0 = np.eye(mdp.daction)
policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0)
# policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0)

#  ======================= HIGH LEVEL SETTINGS =========================  #
n_params = np.size(A0)
mu0 = policy.theta[0:n_params]
Sigma0high = np.eye(n_params)
Sigma0high = Sigma0high + np.diag(np.abs(mu0.ravel()))**2
Sigma0high = nearestSPD(Sigma0high)
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high)

#  ======================== LEARNING SETTINGS ==========================  #
episodes_eval = 1
steps_eval = 100
episodes_learn = 1
steps_learn = 100

# N = 200
# N_MAX = N * 5
# MAXITER = 200
# method = "repsep"
# epsilon = 0.5

N = 200
N_MAX = N * 5
MAXITER = 40
method = "nes"
epsilon = 0.3

checkpoint_path = f"./results/rl_fishing/{method}_{N}_{MAXITER}_0{int(epsilon*10)}_{current_time}"
start = datetime.now()
best_policy, policy_high, hyperv_history = run(N, N_MAX, MAXITER, episodes_learn, steps_learn, policy, policy_high,
                                               mdp, method, epsilon, checkpoint_path, None, r_metric=r_metric)
print(datetime.now() - start)

N_EVAL = 2000
eval(N_EVAL, mdp, episodes_eval, steps_eval, policy, best_policy, policy_high, hyperv_history, None, r_metric=r_metric)