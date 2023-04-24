# import numpy as np
#
#
# def dam_uncertain_problem(
#         env,
#         n_obj,
#         s_init=0,
#         steps=100,
#         **kwargs
# ):
#     decisions = np.array([kwargs[str(i)] for i in range(steps)])
#     reward_1 = 0
#     reward_2 = 0
#     terminal = np.array([False])
#     state = env.ema_reset(s_init)
#     last_action = 0
#
#     for t in range(0, steps):
#         action = decisions[t]
#         nextstate, reward, terminal = env.simulator(state, action, t+1, last_action)
#
#         reward_1 += reward[0]
#         reward_2 += reward[1]
#
#         state = nextstate
#         last_action = action
#
#     if not terminal[0]:
#         raise RuntimeError
#
#     reward_1 /= steps
#     reward_2 /= steps
#
#     if n_obj == 2:
#         return {"upstream_flooding": reward_1[0], "water_demand": reward_2[0],
#                 "upstream_flooding_std": f_std[:, 0], "water_demand_std": f_std[:, 1]}
#     else:
#         raise NotImplementedError
#
