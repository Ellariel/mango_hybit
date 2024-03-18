"""
This file contains the scenario augmentation script. To start it, just run this
script from the command line::

    $ python scalability.py

"""
import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
from _mosaik_components.mas.cells import create_cells, generate_profiles
from _mosaik_components.mas.utils import set_random_seed, get_random_seed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--attempts', default=5, type=int)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--log', default=True, type=bool)
args = parser.parse_args()

base_dir = args.dir
logs_dir = os.path.join(base_dir, 'logs')
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
attempts = args.attempts
stdout_logs = args.log
set_random_seed(seed=args.seed)

NULL = '/dev/null' if os.name == 'posix' else 'nul'

output_filename = 'temp_results.csv'
scalability_time_filename = 'scalability_time.csv'
simulation_time_filename = 'simulation_time.csv'
net_file = os.path.join(data_dir, 'cells.json')
prof_file = os.path.join(data_dir, 'profiles.json')
temp_filename = os.path.join(data_dir, output_filename)
scalability_time_filename = os.path.join(results_dir, scalability_time_filename)
simulation_time_filename = os.path.join(results_dir, simulation_time_filename)

cells_count = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
hierarchy = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(f"cells count: {', '.join([str(i) for i in cells_count])}")
print(f"hierarchy depth: {', '.join([str(i) for i in hierarchy])}")

scalability_time = []
simulation_time = []
for i in tqdm(cells_count):
    net, _ = create_cells(cells_count=i, dir=data_dir)
    for j in tqdm(hierarchy, desc=f"cells_count: {i}", leave=False):
        for _ in tqdm(range(attempts), desc=f"hierarchy: {j}", leave=False):
            seed = get_random_seed()
            generate_profiles(net, dir=data_dir, seed=seed)
            
            start_time = time.time()
            os.system(f"python scenario.py --verbose 1 --seed {seed} --dir {base_dir} --output_file {output_filename} --clean False --cells {i} --hierarchy {j} > {NULL if not stdout_logs else os.path.join(logs_dir, f'stdout_{i}_{j}_{seed}.log')} 2>&1")
            simulation_time += [(i, j, time.time() - start_time)]

            r = pd.read_csv(temp_filename)
            r = r[[c for c in r.columns if 'MosaikAgent-steptime' in c]]
            scalability_time += [(n+1, i, j, k) for n, k in enumerate(r.iloc[:,0].values)]
            os.remove(temp_filename)

scalability_time = pd.DataFrame().from_dict(scalability_time).rename(columns={0: 'step',
                                                            1: 'cells_count',
                                                            2: 'hierarchy_depth',
                                                            3: 'agents_time'})
# scalability_time['per_cell'] = scalability_time['agents_time'] / scalability_time['cells_count']
# scalability_time['per_level'] = scalability_time['agents_time'] / scalability_time['hierarchy_depth']
scalability_time.to_csv(scalability_time_filename, index=False)

simulation_time = pd.DataFrame().from_dict(simulation_time).rename(columns={0: 'cells_count',
                                                                            1: 'hierarchy_depth',
                                                                            2: 'sim_time'})
# simulation_time['per_cell'] = simulation_time['sim_time'] / simulation_time['cells_count']
# simulation_time['per_level'] = simulation_time['sim_time'] / simulation_time['hierarchy_depth']
simulation_time.to_csv(simulation_time_filename, index=False)

print(f"Results were saved to {scalability_time_filename} and {simulation_time_filename}")