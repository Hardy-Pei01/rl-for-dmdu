import numpy as np


def fishing_problem(
        env,
        steps=100,
        **kwargs
):
    daction = env.daction
    decisions = np.array([kwargs[str(i)] for i in range(steps * daction)])
    total_rewards = np.zeros(daction)
    terminal = np.array([False])
    state = env.reset(n=1)
    last_action = 0

    for t in range(0, steps):
        action = decisions[t*daction: t*daction+daction][:, None]
        nextstate, reward, terminal = env.simulator(state, action, t + 1, last_action)

        total_rewards += reward.ravel()

        state = nextstate
        last_action = action

    if not terminal[0]:
        raise RuntimeError

    results = {}
    for i in range(daction):
        results[f'fish_population_{i+1}'] = total_rewards[i]
    return results
