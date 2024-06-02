import numpy as np


def dam_problem(
        env,
        n_obj,
        steps=100,
        **kwargs
):
    decisions = np.array([kwargs[str(i)] for i in range(steps)])
    reward_1 = 0
    reward_2 = 0
    reward_3 = 0
    reward_4 = 0
    terminal = np.array([False])
    state = env.reset(n=1)
    last_action = 0

    for t in range(0, steps):
        action = decisions[t]
        nextstate, reward, terminal = env.simulator(state, action, t+1, last_action)

        reward_1 += reward[0]
        reward_2 += reward[1]
        if n_obj >= 3:
            reward_3 += reward[2]
        if n_obj >= 4:
            reward_4 += reward[3]

        state = nextstate
        last_action = action

    if not terminal[0]:
        raise RuntimeError

    reward_1 /= steps
    reward_2 /= steps
    reward_3 /= steps
    reward_4 /= steps

    if n_obj == 2:
        return {"upstream_flooding": reward_1[0], "water_demand": reward_2[0]}
    elif n_obj == 3:
        return {"upstream_flooding": reward_1[0], "water_demand": reward_2[0], "electricity_demand":reward_3[0]}
    elif n_obj == 4:
        return {"upstream_flooding": reward_1[0], "water_demand": reward_2[0], "electricity_demand":reward_3[0],
                "downstream_flooding": reward_4[0]}
    else:
        raise NotImplementedError
