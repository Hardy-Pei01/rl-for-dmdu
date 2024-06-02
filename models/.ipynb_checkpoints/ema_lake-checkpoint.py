import numpy as np


def lake_problem(
        env,
        steps=99,
        **kwargs
):
    decisions = np.array([kwargs[str(i)] for i in range(steps)])
    reward_1 = 0
    reward_2 = 0
    terminal = np.array([False])
    state = env.reset(n=1)
    last_action = 0

    for t in range(0, steps):
        action = decisions[t]
        nextstate, reward, terminal = env.simulator(state, action, t + 1, last_action)

        reward_1 += reward[0]
        reward_2 += reward[1]

        state = nextstate
        last_action = action

    if not terminal[0]:
        raise RuntimeError

    return {"utility": reward_1[0], "reliability": reward_2[0]}
