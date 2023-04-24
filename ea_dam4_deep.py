from datetime import datetime
from algs.EA_Robust_Trainer import ea_robust_training
from utils.ea.create_models import create_dam4_deep_training
from models.ema_dam_uncertain import dam_uncertain_problem

if __name__ == "__main__":
    for random_seed in [1]:
        current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
        start = datetime.now()
        ea_robust_training(nfe=2000000, random_seed=random_seed, scenario_num=50,
                           model=create_dam4_deep_training,
                           problem=dam_uncertain_problem,
                           scenario_path='./results/dam_scenarios/train_scenarios.csv',
                           restore_path=None, borg=False, n_processes=75,
                           results_path=f'./results/ea_dam_deep/4_avg_200_{random_seed}{current_time_ea}',
                           checkpoint_path=f'./tmp/ea_dam_deep/4_avg_200_{random_seed}{current_time_ea}.pickle',
                           r_metric="avg")
        print("Time duration of training: ", datetime.now() - start)
