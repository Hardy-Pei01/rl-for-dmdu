import copy
import numpy as np
from utils.rl.pareto import pareto


def fishing_problem_rl(
        policy,
        policy_high,
        N_EVAL,
        env,
        steps=100
):
    daction = env.daction

    Theta_eval = policy_high.drawAction(N_EVAL)
    pol_eval = []
    for i in range(0, N_EVAL):
        policy.update(Theta_eval[:, i:i+1])
        pol_eval.append(copy.deepcopy(policy))

    J = np.zeros((daction, N_EVAL))
    terminal = np.array([False])
    state = env.reset(N_EVAL)
    action = np.zeros((daction, N_EVAL))
    last_action = np.zeros((daction, N_EVAL))

    for t in range(0, steps):

        for i in range(0, N_EVAL):
            action[:, i:i+1] = pol_eval[i].drawAction(state[:, i:i+1])
        nextstate, reward, terminal = env.simulator(state, action, t+1, last_action)

        J += reward

        state = nextstate
        last_action = action

    if not terminal[0]:
        raise RuntimeError

    f, _, _ = pareto(J.T, pol_eval)

    results = {}
    for i in range(daction):
        results[f'fish_population_{i+1}'] = f[:, i]
    return results