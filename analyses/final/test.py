import os, sys
sys.path.append(os.path.abspath('../../'))
import pandas as pd
import numpy as np
from EMAWorkbench import load_results
from utils.rl.hv import HyperVolume

def compute_hypervolume(df, antiutopia, utopia):
    array = df.values
    hv_computer = HyperVolume(antiutopia, utopia)
    return hv_computer.compute(array)


results_path='../../results/dam_discrete_performance_2/'
files=sorted(os.listdir(results_path))


ea_hv = []
for i in range(1, 21):
    ea=[]
    ea_files = sorted(os.listdir(results_path + files[i]))
    for j in range(len(ea_files)):
        _, all_outcomes=load_results(results_path + files[i] + "/" + ea_files[j])
        all_outcomes=dict((k, np.mean(all_outcomes[k])) for k in ('upstream_flooding', 'water_demand'))
        ea.append(all_outcomes)
        # print(ea_files[j])
    ea=pd.DataFrame(ea)
    each_ea_hv = compute_hypervolume(ea, np.array([-1, -10]), np.array([0, -9]))
    ea_hv.append(each_ea_hv)
    print(files[i])
    # print(ea)
    print(each_ea_hv)
ea_hv = np.array(ea_hv)
print(ea_hv.mean())
print(ea_hv.std())