import numpy as np
import scipy as sp


def percentile_10(outcomes):
    outcome_1 = outcomes["utility"]
    score_1 = np.percentile(outcome_1, q=10)

    outcome_2 = outcomes["reliability"]
    score_2 = np.percentile(outcome_2, q=10)

    return score_1, score_2


def avg(outcomes):
    outcome_1 = outcomes["utility"]
    score_1 = np.mean(outcome_1)

    outcome_2 = outcomes["reliability"]
    score_2 = np.mean(outcome_2)

    return score_1, score_2


def mean_variance(outcomes):
    outcome_1 = outcomes["utility"]
    mean_1 = np.mean(outcome_1)
    std_1 = np.std(outcome_1)
    score_1 = (mean_1 + 1) / (std_1 + 1)

    outcome_2 = outcomes["reliability"]
    mean_2 = np.mean(outcome_2)
    std_2 = np.std(outcome_2)
    score_2 = (mean_2 + 1) / (std_2 + 1)

    return score_1, score_2

def obj3_avg(outcomes):
    outcome_1 = outcomes["upstream_flooding"]
    score_1 = np.mean(outcome_1)

    outcome_2 = outcomes["water_demand"]
    score_2 = np.mean(outcome_2)
    
    outcome_3 = outcomes["electricity_demand"]
    score_3 = np.mean(outcome_3)

    return score_1, score_2, score_3


def obj_func(outcomes):
    outcome_1 = outcomes["utility"]
    q1 = sp.stats.scoreatpercentile(outcome_1, per=25)
    median = sp.stats.scoreatpercentile(outcome_1, per=50)
    q3 = sp.stats.scoreatpercentile(outcome_1, per=75)

    dispersion = abs(q3 - q1)
    score_1 = median * (dispersion + 1)

    outcome_2 = outcomes["reliability"]
    q1 = sp.stats.scoreatpercentile(outcome_2, per=25)
    median = sp.stats.scoreatpercentile(outcome_2, per=50)
    q3 = sp.stats.scoreatpercentile(outcome_2, per=75)

    dispersion = abs(q3 - q1)
    score_2 = median * (dispersion + 1)

    return score_1, score_2
