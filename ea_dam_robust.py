from datetime import datetime
from algs.EA_Robust_Trainer import ea_robust_training
from utils.ea.create_models import create_dam_robust_training
from models.ema_dam_uncertain import dam_uncertain_problem

if __name__ == "__main__":
    r_metric = "avg"
    for random_seed in [3186775264, 3690172787, 462638671, 1926216712, 3087161096]:
        current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
        start = datetime.now()
        nfe = 30000
        ea_robust_training(nfe=nfe, random_seed=random_seed, scenario_num=50,
                           model=create_dam_robust_training,
                           problem=dam_uncertain_problem,
                           scenario_path='./results/dam_scenarios/train_scenarios.csv',
                           restore_path=None, borg=False, n_processes=75,
                           results_path=f'./results/ea_dam_robust/{r_metric}_{nfe//10000}_{random_seed}{current_time_ea}',
                           checkpoint_path=f'./tmp/ea_dam_robust/{r_metric}_{nfe//10000}_{random_seed}{current_time_ea}',
                           r_metric=r_metric, n_obj=2)
        print("Time duration of training: ", datetime.now() - start)
