import numpy as np
import pandas as pd
import copy


def evaluate_policies(mdp, episodes, maxsteps, policies, scenarios, r_metric):
    # EVALUATE_POLICIES Evaluates a set of policies. For each policy, several
    # episodes are simulated.
    # The function is very similar to COLLECT_SAMPLES, but it accepts many
    # policies as input and it does not return the low level dataset.

    #    INPUT
    #     - mdp      : the MDP to be solved
    #     - episodes : number of episodes per policy
    #     - maxsteps : max number of steps per episode
    #     - policy   : policies to be evaluated
    #     - contexts : (optional) contexts of each episode

    #    OUTPUT
    #     - J        : returns of each policy

    npolicy = np.asarray(policies).size
    totepisodes = episodes * npolicy
    # Initialize variables
    J = np.zeros((mdp.dreward, totepisodes))
    step = 0
    # Initialize simulation
    if scenarios is None:
        state = mdp.reset(totepisodes)
    else:
        scenarios = pd.concat([scenarios] * npolicy, ignore_index=True)
        state = mdp.manifold_reset(scenarios)
    action = np.zeros((mdp.daction, totepisodes))
    last_action = np.zeros((mdp.daction, totepisodes))
    # Keep track of the states which did not terminate
    ongoing = np.ones(totepisodes)

    # Save the last step per episode
    endingstep = maxsteps * np.ones(totepisodes)
    # Run the episodes until maxsteps or all ends
    while (step < maxsteps) and np.sum(ongoing) > 0:

        step = step + 1
        # Select action
        for i in range(0, npolicy):
            idx = np.arange(i * episodes, i * episodes + episodes)
            doStates = state[:, idx[ongoing[idx] == 1]]
            if doStates.shape[1] != 0:
                action[:, idx[ongoing[idx] == 1]] = policies[i].drawAction(state[:, idx[ongoing[idx] == 1]])
        # Simulate one step of all running episodes at the same time
        nextstate, reward, terminal = mdp.simulator(state[:, ongoing == 1], copy.deepcopy(action[:, ongoing == 1]),
                                                    step, copy.deepcopy(last_action[:, ongoing == 1]))

        state[:, ongoing == 1] = nextstate
        # Update the total reward
        J[:, ongoing == 1] = J[:, ongoing == 1] + np.multiply(mdp.gamma ** (step - 1), reward)
        # Continue
        idx = np.arange(0, totepisodes)
        idx = idx[ongoing == 1]
        idx = idx[terminal == 1]
        endingstep[idx] = step
        ongoing[ongoing == 1] = terminal == 0
        last_action = copy.deepcopy(action)

    # If we are in the average reward setting, then normalize the return
    if mdp.isAveraged and mdp.gamma == 1:
        J = J * (1 / endingstep)

    if r_metric == "10th":
        J = np.percentile(J.reshape((mdp.dreward, npolicy, episodes)), q=10, axis=2)
    elif r_metric == "mv":
        J_reshape = J.reshape((mdp.dreward, npolicy, episodes))
        mean = J_reshape.mean(axis=2)
        std = J_reshape.std(axis=2)
        J = (mean + 1) / (std + 1)
    elif r_metric == "avg":
        J = J.reshape((mdp.dreward, npolicy, episodes)).mean(axis=2)
    else:
        raise NotImplementedError
    return J
