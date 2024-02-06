"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import sys
import os
import time
from tqdm import tqdm
import json
import pandas as pd

base_dir = './'
output_filename = 'temp_results.csv'
scalability_time_filename = 'scalability_time.csv'
simulation_time_filename = 'simulation_time.csv'
cells_count = list(range(2, 16+1))


logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(logs_dir, exist_ok=True)
temp_filename = os.path.join(base_dir, output_filename)
scalability_time_filename = os.path.join(base_dir, scalability_time_filename)
simulation_time_filename = os.path.join(base_dir, simulation_time_filename)
print(f"cells_count: {', '.join([str(i) for i in cells_count])}")

scalability_time = []
simulation_time = []
for i in tqdm(cells_count):
    start_time = time.time()
    os.system(f"python scenario.py --dir {base_dir} --output_file {output_filename} --clean True --cells {i} >> {os.path.join(logs_dir, f'stdout_{i}.log')}")
    simulation_time += [(i, time.time() - start_time)]

    r = pd.read_csv(temp_filename)
    r = r[[c for c in r.columns if 'MosaikAgent-steptime' in c]]
    scalability_time += [(n+1, i, j) for n, j in enumerate(r.iloc[:,0].values)]
    os.remove(temp_filename)

scalability_time = pd.DataFrame().from_dict(scalability_time).rename(columns={0: 'step',
                                                            1: 'cells_count',
                                                            2: 'step_time'})
scalability_time['per_cell'] = scalability_time['step_time'] / scalability_time['cells_count']
scalability_time.to_csv(scalability_time_filename, index=False)

simulation_time = pd.DataFrame().from_dict(simulation_time).rename(columns={0: 'cells_count',
                                                            1: 'sim_time'})
simulation_time['per_cell'] = simulation_time['sim_time'] / simulation_time['cells_count']
simulation_time.to_csv(simulation_time_filename, index=False)

print(f"Results were saved to {scalability_time_filename} and {simulation_time_filename}")