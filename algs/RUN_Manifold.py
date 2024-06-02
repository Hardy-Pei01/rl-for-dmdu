import copy
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from datetime import datetime
from algs.NES_Solver import NES_Solver
from algs.REPSep_Solver import REPSep_Solver
from utils.rl.hypervolume2d import hypervolume2d
from utils.rl.hypervolume import hypervolume
from utils.rl.hv import HyperVolume
from utils.rl.metric_hv import metric_hv
from utils.rl.evaluate_policies import evaluate_policies
from utils.rl.mixtureIS import mixtureIS
from utils.rl.pareto import pareto


# def hyperv(dreward, J, utopia, antiutopia):
#     if dreward == 2:
#         return hypervolume2d(J, antiutopia, utopia)
#     else:
#         return hypervolume(J, antiutopia, utopia, 50000)


def hyperv(J, hv_computer):
    return hv_computer.compute(J)


# def metric(dreward, J, utopia, antiutopia):
#     return metric_hv(J, lambda x: hyperv(dreward, x, utopia, antiutopia))


def metric(J, hv_computer):
    return metric_hv(J, lambda x: hyperv(x, hv_computer))


def run(N, N_MAX, MAXITER, episodes_learn, steps_learn, policy, policy_high,
        mdp, method, epsilon, checkpoint_path, scenarios, r_metric):
    if method == "nes":
        solver = NES_Solver(epsilon)
        distance = 'Norm'
    elif method == "repsep":
        solver = REPSep_Solver(epsilon)
        distance = 'KL Div'
    else:
        raise NotImplementedError

    policy.makeDeterministic()

    utopia = mdp.utopia
    antiutopia = mdp.antiutopia
    hv_computer = HyperVolume(antiutopia, utopia)

    iter = 1
    # MAXITER = MAXEPISODES / N / episodes_learn
    best_policy = None
    high_policy = None
    best_hv = 0
    high_hv = 0
    hyperv_history = []

    ## Learning
    while iter <= MAXITER:

        # Draw N policies and evaluate them
        Theta_iter = policy_high.drawAction(N)
        p = np.array([copy.deepcopy(policy) for _ in range(N)])
        for i in range(N):
            p[i].update(Theta_iter[:, i:i + 1])
        J_iter = evaluate_policies(mdp, episodes_learn, steps_learn, p, scenarios, r_metric)
        # First, fill the pool to maintain the samples distribution
        if iter == 1:
            J = np.tile(np.amin(J_iter, 1), (N_MAX, 1)).T
            Theta = policy_high.drawAction(N_MAX)
            Policies = np.array([copy.deepcopy(policy_high) for _ in range(N_MAX)])
        # Enqueue the new samples and remove the old ones
        J = np.concatenate((J_iter, J[:, 0:(N_MAX - N)]), axis=1)
        Theta = np.concatenate((Theta_iter, Theta[:, 0:(N_MAX - N)]), axis=1)
        Policies = np.concatenate((np.array([copy.deepcopy(policy_high) for _ in range(N)]), Policies[0:(N_MAX - N)]))
        # Compute IS weights
        W = mixtureIS(policy_high, Policies, N, Theta)
        #     W = ones(1,N_MAX);
        # Scalarize samples
        fitness = metric(J.T, hv_computer).T
        # Perform an update step
        div = solver.step(fitness, Theta, policy_high, W)
        # print(iter)
        if div < 1e-06:
            continue
        hv = hyperv(pareto(J.T)[0], hv_computer)
        hv_iter = hyperv(pareto(J_iter.T)[0], hv_computer)
        print('Iter: %d, Hyperv: %.4f, %s: %.2f\n' % (iter, hv, distance, div))
        print('Iter: %d, Hyperv: %.4f\n' % (iter, hv_iter))
        hyperv_history.append(hv)
        iter = iter + 1

        if hv_iter >= best_hv:
            best_policy = copy.deepcopy(policy_high)
            best_hv = hv_iter
            print("!!!")

        if hv >= high_hv:
            high_policy = copy.deepcopy(policy_high)
            high_hv = hv
            print("???")

    policy_high_path = f'{checkpoint_path}_high.pickle'
    policy_best_path = f'{checkpoint_path}_best.pickle'
    with open(policy_high_path, 'wb') as fh:
        pickle.dump(high_policy, fh)
    with open(policy_best_path, 'wb') as fh:
        pickle.dump(best_policy, fh)

    return best_policy, high_policy, hyperv_history


def eval(N_EVAL, mdp, episodes_eval, steps_eval, policy, best_policy, policy_high, hyperv_history, scenarios, r_metric):
    hv_computer = HyperVolume(mdp.antiutopia, mdp.utopia)
    current_time = datetime.now().strftime("-%d-%m-%Y-%H-%M")
    Theta_eval = best_policy.drawAction(N_EVAL)
    pol_eval = []
    for i in range(0, N_EVAL):
        policy.update(Theta_eval[:, i:i+1])
        pol_eval.append(copy.deepcopy(policy))

    f_eval = np.transpose(evaluate_policies(mdp, episodes_eval, steps_eval, pol_eval, scenarios, r_metric))
    f, p, _, _ = pareto(f_eval, pol_eval)
    best_hyper = hyperv(f, hv_computer)
    # pd.DataFrame(f).to_csv(f'./results/{problem}_{method}_best_{current_time}.csv', index=False)
    print('Hypervolume: %.4f\n' % best_hyper)
    # print(max(f[:, 0]), max(f[:, 1]), max(f[:, 2]), max(f[:, 3]))
    # print(min(f[:, 0]), min(f[:, 1]), min(f[:, 2]), min(f[:, 3]))
    print(f)

    # Theta_eval = policy_high.drawAction(N_EVAL)
    # pol_eval = []
    # for i in range(0, N_EVAL):
    #     policy.update(Theta_eval[:, i:i+1])
    #     pol_eval.append(copy.deepcopy(policy))
    #
    # f_eval = np.transpose(evaluate_policies(mdp, episodes_eval, steps_eval, pol_eval, scenarios, r_metric))
    # f, p, _, _ = pareto(f_eval, pol_eval)
    # high_hyper = hyperv(f, hv_computer)
    # # pd.DataFrame(f).to_csv(f'./results/{problem}_{method}_high_{current_time}.csv', index=False)
    # print('Hypervolume: %.4f\n' % high_hyper)
    # # print(max(f[:, 0]), max(f[:, 1]), max(f[:, 2]), max(f[:, 3]))
    # # print(min(f[:, 0]), min(f[:, 1]), min(f[:, 2]), min(f[:, 3]))
    # print(f)

    # if hyperv_history is not None:
    #     plt.plot(hyperv_history)
    #     plt.show()

    return best_hyper, f
