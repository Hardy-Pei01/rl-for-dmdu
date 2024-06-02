import numpy as np


def lake4_uncertain_problem(
        env,
        b=0.42,  # decay rate for P in lake (0.42 = irreversible)
        q=2.0,  # recycling exponent
        mean=0.02,  # mean of natural inflows
        stdev=0.0017,  # future utility discount rate
        delta=0.98,  # standard deviation of natural inflows
        steps=100,
        **kwargs
):
    decisions = np.array([kwargs[str(i)] for i in range(steps)])
    reward_1 = 0
    reward_2 = 0
    reward_3 = 0
    reward_4 = 0
    terminal = np.array([False])
    state = env.ema_reset(b, q, mean, stdev, delta)
    last_action = 0

    for t in range(0, steps):
        action = decisions[t]
        nextstate, reward, terminal = env.simulator(state, action, t + 1, last_action)

        reward_1 -= reward[0]
        reward_2 += reward[1]
        reward_3 += reward[2]
        reward_4 += reward[3]

        state = nextstate
        last_action = action

    if not terminal[0]:
        raise RuntimeError

    return {"avg_pollution": reward_1[0], "utility": reward_2[0],
            "inertia": reward_3[0], "reliability": reward_4[0]}
