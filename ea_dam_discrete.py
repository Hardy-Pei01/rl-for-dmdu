from datetime import datetime
from algs.EA_Trainer import ea_training
from utils.ea.create_models import create_dam_discrete_training
from models.ema_dam import dam_problem

if __name__ == "__main__":
    for random_seed in [3186775264, 3690172787, 462638671, 1926216712, 3087161096,
                        956500800, 2523200676, 1274569960, 2097286424, 3885705317,
                        562732020, 2861224539, 1350287007, 674137616, 3624030427,
                        703574460, 1883682950, 617160326, 3668976038, 96930842]:
        current_time_ea = datetime.now().strftime("-%d-%m-%Y-%H-%M")
        start = datetime.now()
        nfe = 20000
        ea_training(nfe=nfe, random_seed=random_seed,
                    model=create_dam_discrete_training,
                    problem=dam_problem,
                    restore_path=None, borg=False, n_processes=75,
                    results_path=f'./results/ea_dam_1/discrete_{nfe//10000}_{random_seed}{current_time_ea}',
                    checkpoint_path=f'./tmp/ea_dam_1/discrete_{nfe//10000}_{random_seed}{current_time_ea}.pickle')
        print("Time duration of training: ", datetime.now() - start)
