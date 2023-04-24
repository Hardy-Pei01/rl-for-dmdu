from datetime import datetime
from algs.EA_Trainer import ea_training
from utils.ea.create_models import create_dam4_discrete_training
from models.ema_dam import dam_problem

if __name__ == "__main__":
    for random_seed in [1]:
        current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
        start = datetime.now()
        ea_training(nfe=200000, random_seed=random_seed,
                    model=create_dam4_discrete_training,
                    problem=dam_problem,
                    restore_path=None, borg=False, n_processes=75,
                    results_path=f'./results/ea_dam/4_20_{random_seed}{current_time_ea}',
                    checkpoint_path=f'./tmp/ea_dam/4_20_{random_seed}{current_time_ea}.pickle')
        print("Time duration of training: ", datetime.now() - start)
