"""
This file contains the mosaik scenario. To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import os
import sys
import copy
import mosaik
import arrow
import argparse
import mosaik.util
import numpy as np
import pandas as pd
import pandapower as pp
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from massca.lib import Default

parser = argparse.ArgumentParser()
parser.add_argument('--cells', default=2, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--output_file', default='results.csv', type=str)
parser.add_argument('--performance', default=True, type=bool)
parser.add_argument('--between', default='cohda', type=str)
parser.add_argument('--within', default='cohda', type=str)
args = parser.parse_args()

base_dir = args.dir
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
grid_file = os.path.join(data_dir, 'grid_model.json')
prof_file = os.path.join(data_dir, 'grid_profiles.csv')
profiles = pd.read_csv(prof_file, skiprows=1)
profiles.index = profiles["Time"].apply(lambda x: (arrow.get(x) - arrow.get(profiles["Time"].iloc[0])).seconds)
grid = pp.from_json(grid_file)
pp.runpp(grid, numba=False)
print(f"Grid model of {len(grid.load)} loads, {len(grid.sgen)} sgens, {len(grid.bus)} buses, {len(grid.line)} lines, {len(grid.trafo)} trafos")

START_DATE = '2016-01-01 00:00:00'
DATE_FORMAT = 'mixed' #'YYYY-MM-DD hh:mm:ss'
STEP_SIZE = 60*15
END = 60*15 + STEP_SIZE
MAX_EXT_GRID_CAPACITY = 25

SIM_CONFIG = {
    'MASSCA': {
        'python': 'massca:MosaikAgents',
    },
    'FlexSim': {
        'python': 'simulators.flexible:Simulator',
    },
    'GridSim': {
         'python': 'mosaik_components.pandapower:Simulator',
    },
    'OutputSim': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    'InputSim': {
        'python': 'mosaik_csv:CSV'
    },  
}

default = Default()

def get_unit_profile(eid, time_step, profiles):
    eid = eid.split('.')[0]
    idx = profiles.index.get_indexer([time_step], method='ffill')[0]
    item = profiles.iloc[idx]
    return {i : item[f"{eid}.{i}"] 
                for i in ['min', 'max']
                    if f"{eid}.{i}" in item} 

def input_to_state(aeid, aid, input_data, current_state, current_time, first_time_step, **kwargs):
    global profiles, default
    state = copy.deepcopy(default.config['state_dict'])
    for eid, value in input_data.items():
        value = sum(value.values())
        if pd.isna(value):
            value = 0
        if 'Load' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption'] = default.update_flexibility(state['consumption'], profile)
                state['consumption']['level'] = value
        elif 'StaticGen' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = default.update_flexibility(state['production'], profile)
                state['production']['level'] = value
        elif 'ExternalGrid' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                if not len(profile):
                    profile['min'] = 0
                    profile['max'] = MAX_EXT_GRID_CAPACITY
                state['production'] = default.update_flexibility(state['production'], profile)
                state['consumption'] = default.update_flexibility(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['level'] = value
                else:
                    state['consumption']['level'] = abs(value) 
        break 
#
    state['consumption']['delta'] = current_state['consumption']['delta']
    state['production']['delta'] = current_state['production']['delta']
    return state

# Multi-agent system (MAS) configuration
# User-defined methods are specified in massca.lib.default to make scenario concise, 
# except `input_method`, since it requieres an access to global variables
default.build(performance=args.performance,
              input_method=input_to_state,
              between_cells=args.between, #'default', 'cohda'
              within_cell=args.within,
              verbose=args.verbose,
              seed=args.seed, 
              )

world = mosaik.World(SIM_CONFIG)
with world.group():
        massim = world.start('MASSCA', 
                            step_size=STEP_SIZE, 
                            **default.config,
                            )
        flsim = world.start(
                        "FlexSim",
                        step_size=STEP_SIZE,
                        sim_params=dict(gen_neg=False),
                    )
    
input_sim = world.start("InputSim", 
                              sim_start=START_DATE, 
                              date_format=DATE_FORMAT,
                              datafile=prof_file)
inputs = input_sim.Profiles.create(1)[0]
    
output_sim = world.start('OutputSim', start_date = START_DATE,
                                          output_file=os.path.join(results_dir, 
                                                                   f'{args.between}_{args.within}_{args.output_file}'))
outputs = output_sim.CSVWriter(buff_size=STEP_SIZE)
    
grid_sim = world.start('GridSim', step_size=STEP_SIZE)
units = pd.concat([grid.load, grid.sgen, grid.ext_grid, grid.bus], 
                        ignore_index=True).set_index('name')
grid = grid_sim.Grid(json=grid_file)
units = {k : (e, v.to_dict()) for k, v in units.iterrows()
                            for e in grid.children
                                if e.eid == k}

agents = [] # one simple agent per power unit (sgen, load)
controllers = [] # top-level agents that are named as cell_controllers, one per cell
hierarchical_controllers = []

mosaik_agent = massim.MosaikAgents() # core agent for the mosaik communication
world.connect(units['ExternalGrid-0'][0], outputs, ("P[MW]", "value"))
world.connect(units['ExternalGrid-0'][0], mosaik_agent, ('P[MW]', 'ExternalGrid-0.value'), 
                                            time_shifted=True, initial_data={'P[MW]': 0})
world.connect(mosaik_agent, outputs, ("production[MW]", "ExternalGrid-0.production[MW]"))
world.connect(mosaik_agent, outputs, ("consumption[MW]", "ExternalGrid-0.consumption[MW]"))
world.connect(mosaik_agent, outputs, 'steptime')

controllers += massim.MosaikAgents.create(num=2, controller=None)
hierarchical_controllers = [massim.MosaikAgents.create(num=1, controller=i.eid)[0] for i in controllers]

switch_off = [] 
for k, v in units.items():
        cell_idx = int(int(k.split('-')[-1]) % 2 != 0)
        if "Load" in k:
            fload = flsim.FLSim.create(num=1)[0]
            agents += massim.MosaikAgents.create(num=1, controller=hierarchical_controllers[cell_idx].eid)
            world.connect(inputs, fload, (f"{k}.value", "P[MW]"))
            world.connect(fload, agents[-1], ('P[MW]', f"{k}.consumption[MW]"))
            world.connect(agents[-1], fload, ('consumption_delta[MW]', 'scale_factor'), weak=True)
            world.connect(agents[-1], units[f"Bus-{int(v[1]['bus'])}"][0], ("consumption[MW]", "P_load[MW]"))
            world.connect(agents[-1], outputs)
            switch_off.append(k)
            
        elif "StaticGen" in k:
            fgen = flsim.FLSim.create(num=1)[0]
            agents += massim.MosaikAgents.create(num=1, controller=hierarchical_controllers[cell_idx].eid)
            world.connect(inputs, fgen, (f"{k}.value", "P[MW]"))
            world.connect(fgen, agents[-1], ('P[MW]', f"{k}.production[MW]"))
            world.connect(agents[-1], fgen, ('production_delta[MW]', 'scale_factor'), weak=True)
            world.connect(agents[-1], units[f"Bus-{int(v[1]['bus'])}"][0], ("production[MW]", "P_gen[MW]"))
            world.connect(agents[-1], outputs)      
            switch_off.append(k)
            
        elif "Bus" in k:
            world.connect(v[0], outputs, ("P[MW]", "value"))
            pass
            
grid_sim.disable_elements(switch_off)
    
if args.performance:
    mosaik.util.plot_dataflow_graph(world, hdf5path=os.path.join(results_dir, '.hdf5'), show_plot=False)
world.run(until=END, print_progress='individual' if args.performance else True)