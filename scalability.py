"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import sys
import os
import time
from tqdm import tqdm
import pandas as pd
from cells import create_cells, generate_profiles
from _mosaik_components.mas.utils import set_random_seed, get_random_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def hard_close_descriptors():
    KEEP_FD = set([0, 1, 2])
    for fd in os.listdir(os.path.join("/proc", str(os.getpid()), "fd")):
        print(fd)
        if int(fd) not in KEEP_FD:
            try:
                os.close(int(fd))
            except OSError:
                pass

base_dir = './'
attempts = 5
stdout_logs = False
set_random_seed(seed=13)

output_filename = 'temp_results.csv'
scalability_time_filename = 'scalability_time.csv'
simulation_time_filename = 'simulation_time.csv'
logs_dir = os.path.join(base_dir, 'logs')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

net_file = os.path.join(base_dir, 'cells.json')
prof_file = os.path.join(base_dir, 'profiles.json')
temp_filename = os.path.join(base_dir, output_filename)
scalability_time_filename = os.path.join(results_dir, scalability_time_filename)
simulation_time_filename = os.path.join(results_dir, simulation_time_filename)

cells_count = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
hierarchy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(f"cells count: {', '.join([str(i) for i in cells_count])}")
print(f"hierarchy depth: {', '.join([str(i) for i in hierarchy])}")

scalability_time = []
simulation_time = []
for i in tqdm(cells_count):
    net, _ = create_cells(cells_count=i, dir=base_dir)
    for j in tqdm(hierarchy, desc=f"cells_count: {i}"):
        for _ in tqdm(range(attempts), desc=f"hierarchy: {j}"):
            seed = get_random_seed()
            generate_profiles(net, dir=base_dir, seed=seed)
            
            start_time = time.time()
            os.system(f"python scenario.py --seed {seed} --dir {base_dir} --output_file {output_filename} --clean False --cells {i} --hierarchy {j} >> {'nul' if not stdout_logs else os.path.join(logs_dir, f'stdout_{i}_{seed}.log')}")
            simulation_time += [(i, j, time.time() - start_time)]

            # time.sleep(1) # wait as we are not sure if os.system closes all the file descriptors before we use it

            r = pd.read_csv(temp_filename)
            r = r[[c for c in r.columns if 'MosaikAgent-steptime' in c]]
            scalability_time += [(n+1, i, j, k) for n, k in enumerate(r.iloc[:,0].values)]
            os.remove(temp_filename)

            hard_close_descriptors()

            # time.sleep(1) # wait as we are not sure if os.system deletes files on time

scalability_time = pd.DataFrame().from_dict(scalability_time).rename(columns={0: 'step',
                                                            1: 'cells_count',
                                                            2: 'hierarchy_depth',
                                                            3: 'agents_time'})
scalability_time['per_cell'] = scalability_time['agents_time'] / scalability_time['cells_count']
scalability_time['per_level'] = scalability_time['agents_time'] / scalability_time['hierarchy_depth']
scalability_time.to_csv(scalability_time_filename, index=False)

simulation_time = pd.DataFrame().from_dict(simulation_time).rename(columns={0: 'cells_count',
                                                                            1: 'hierarchy_depth',
                                                                            2: 'sim_time'})
simulation_time['per_cell'] = simulation_time['sim_time'] / simulation_time['cells_count']
simulation_time['per_level'] = simulation_time['sim_time'] / simulation_time['hierarchy_depth']
simulation_time.to_csv(simulation_time_filename, index=False)

print(f"Results were saved to {scalability_time_filename} and {simulation_time_filename}")