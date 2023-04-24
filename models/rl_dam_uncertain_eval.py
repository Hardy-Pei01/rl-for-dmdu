import copy
import numpy as np
import pandas as pd
from utils.rl.pareto import pareto


def dam_uncertain_problem_rl_eval(
        policy,
        policy_high,
        episodes,
        N_EVAL,
        env,
        scenarios,
        n_obj=2,
        steps=100
):
    Theta_eval = policy_high.drawAction(N_EVAL)
    pol_eval = []
    for i in range(0, N_EVAL):
        policy.update(Theta_eval[:, i:i+1])
        pol_eval.append(copy.deepcopy(policy))
    
    totepisodes = episodes * N_EVAL
    J = np.zeros((n_obj, totepisodes))
    terminal = np.array([False])
    scenarios = pd.concat([scenarios] * N_EVAL, ignore_index=True)
    state = env.manifold_reset(scenarios)
    action = np.zeros((1, totepisodes))
    last_action = np.zeros((1, totepisodes))

    for t in range(0, steps):

        for i in range(0, N_EVAL):
            idx = np.arange(i * episodes, i * episodes + episodes)
            action[:, idx] = pol_eval[i].drawAction(state[:, idx])

        nextstate, reward, terminal = env.simulator(state, action, t+1, last_action)

        J += reward
        state = nextstate
        last_action = action


    if not terminal[0]:
        raise RuntimeError

    J /= steps
    J_std = J.reshape((n_obj, N_EVAL, episodes)).std(axis=2)
    J = J.reshape((n_obj, N_EVAL, episodes)).mean(axis=2)

    f, _, f_std, _ = pareto(J.T, pol_eval, J_std.T)

    if n_obj == 2:
        return {"upstream_flooding": f[:, 0], "water_demand": f[:, 1],
                "upstream_flooding_std": f_std[:, 0], "water_demand_std": f_std[:, 1]}
    else:
        raise NotImplementedError