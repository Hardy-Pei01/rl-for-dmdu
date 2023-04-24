from datetime import datetime
from algs.EA_Trainer import ea_training
from utils.ea.create_models import create_fishing_training
from models.ema_fishing import fishing_problem

if __name__ == "__main__":
    for random_seed in [1]:
        current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
        start = datetime.now()
        ea_training(nfe=2000000, random_seed=random_seed,
                    model=create_fishing_training,
                    problem=fishing_problem,
                    borg=False, n_processes=75,
                    results_path=f'./results/ea_fishing/200_{random_seed}{current_time_ea}',
                    restore_path=None, checkpoint_path=f'./tmp/ea_fishing/{random_seed}{current_time_ea}.pickle')
        print("Time duration of training: ", datetime.now() - start)
