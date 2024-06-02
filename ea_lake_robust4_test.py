import warnings
import pandas as pd
from datetime import datetime
from algs.EA_Robust_Trainer4 import ea_robust_training4
from utils.ea.create_models import create_lake4_deep_training_test
from models.ema_lake4_uncertain_test import lake4_uncertain_problem_test

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    for random_seed in [
                        3186775264,
                        # 3690172787, 462638671, 1926216712, 3087161096,
                        # 956500800, 2523200676, 1274569960, 2097286424, 3885705317,
                        # 562732020, 2861224539, 1350287007, 674137616, 3624030427,
                        # 703574460, 1883682950, 617160326, 3668976038, 96930842
    ]:
        for r_metric in ["10th", "avg"]:
            current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
            start = datetime.now()
            nfe = 10000
            ea_robust_training4(nfe=nfe, random_seed=random_seed, scenario_num=25,
                                model=create_lake4_deep_training_test,
                                problem=lake4_uncertain_problem_test,
                                scenario_path='./results/lake_scenarios/train_scenarios.csv',
                                restore_path=None, borg=False, n_processes=4,
                                results_path=f'./results/ea_lake_robust_test/{r_metric}_{nfe//10000}_{random_seed}{current_time_ea}',
                                checkpoint_path=f'./tmp/ea_lake_robust_test/{r_metric}_{nfe//10000}_{random_seed}{current_time_ea}',
                                r_metric=r_metric, n_obj=4)
            print("Time duration of training: ", datetime.now() - start)
