"""
This file contains the mosaik scenario. To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import sys
import os
import json
import mosaik
import random
import argparse
import numpy as np
import more_itertools as mit
import pandapower as pp
import pandapower.networks as pn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

from _mosaik_components.mas.cells import create_cells, generate_profiles
from _mosaik_components.mas.utils import set_random_seed, check_file_descriptors
from _mosaik_components.mas.mosaik_agents import *
from _mosaik_components.mas.methods import *

parser = argparse.ArgumentParser()
parser.add_argument('--cells', default=2, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--clean', default=True, type=bool)
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--output_file', default='results.csv', type=str)
parser.add_argument('--performance', default=True, type=bool)
parser.add_argument('--hierarchy', default=1, type=int)
args = parser.parse_args()

base_dir = args.dir
cells_count = args.cells
hierarchy = args.hierarchy

print(f"cells count: {cells_count}")
print(f"hierarchy depth: {hierarchy}")

# loading network and flexibility profiles
cells = {}

net_file = os.path.join(base_dir, 'cells.json')
if not os.path.exists(net_file) or args.clean:
    net, net_file = create_cells(cells_count=cells_count)
else:
    net = pp.from_json(net_file)

print(f"loads: {len(net.load)}, gens: {len(net.sgen)}")

prof_file = os.path.join(base_dir, 'profiles.json')
if not os.path.exists(prof_file) or args.clean:
    profiles, prof_file = generate_profiles(net, seed=args.seed)
else:
    with open(prof_file, 'r') as f:
        profiles = json.load(f)

END = 3600 * 1 #24 * 1  # 1 day
START_DATE = '2014-01-01 12:00:00'
GRID_FILE = net_file
WIND_FILE = 'demo/wind_speed_m-s_15min.csv'
STEP_SIZE = 60 * 15

# simulators
SIM_CONFIG = {
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator' #2
         # 'python': 'mosaik_pandapower.simulator:Pandapower'
    },
    'PVSim': {
        'python': 'mosaik_components.pv.pvgis_simulator:PVGISSimulator',
    },
    'FLSim': {
        'python': '_mosaik_components.flexible.flexiload_simulator:FLSimulator',
    },
    'CSV': {
        'python': 'mosaik_csv:CSV',
    },
    'WecsSim': {
        'python': '_mosaik_components.wecssim.mosaik:WecsSim',
    },
    'MAS': {
        'python': '_mosaik_components.mas.mosaik_agents:MosaikAgents',
    },
    'CSV_writer': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    'InputSim': {
        'python': '_mosaik_components.basic_simulators.input_simulator:InputSimulator',
    },
}

# PV simulator
PVSIM_PARAMS = {
    'start_date' : START_DATE,
    'cache_dir' : False, #'./', # it caches PVGIS API requests
    'verbose' : False, # print PVGIS parameters and requests
    'gen_neg' : False, # return negative P
}

# For each PV
PVMODEL_PARAMS = {
    'scale_factor' : 100,
    'lat' : 52.373,
    'lon' : 9.738,
    'optimal_both' : True,
}

# Wind power
WECS_CONFIG = {'P_rated': 7000, 
               'v_rated': 13, 
               'v_min': 3.5, 
               'v_max': 25}

# method that transforms mosaik inputs dict to the agent state (called for each agent/connected unit, default: copy dict)
def input_to_state(input_data, current_state):
    # state={'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0}}
    global cells
    state = copy.deepcopy(current_state)
    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            profile = get_unit_profile(eid, cells)
            scale_factor = input_data.get('scale_factor', 1)
            if 'Load' in eid or 'FL' in eid: # check the type of connected unit and its profile
                state['consumption']['min'] = profile['min']
                state['consumption']['max'] = profile['max']
                if scale_factor == 1 and value == 0:
                    value = random.uniform(profile['min'], profile['max'])
                state['consumption']['current'] = np.clip(abs(value), profile['min'], profile['max'])
            elif 'Gen' in eid or 'PV' in eid:
                state['production']['min'] = profile['min']
                state['production']['max'] = min(abs(value), profile['max'])
                state['production']['current'] = np.clip(abs(value), profile['min'], min(abs(value), profile['max']))
            elif 'ExternalGrid' in eid:
                if value >= 0: # check the convention here!
                    state['production']['current'] = value
                    state['production']['min'] = value * -3
                    state['production']['max'] = value * 3
                    state['consumption']['current'] = 0
                    state['consumption']['min'] = 0
                    state['consumption']['max'] = 0
                else:
                    state['production']['current'] = 0
                    state['production']['min'] = 0
                    state['production']['max'] = 0
                    state['consumption']['current'] = abs(value)
                    state['consumption']['min'] = abs(value) * -3
                    state['consumption']['max'] = abs(value) * 3
    return state

# Multi-agent system (MAS) configuration
# User-defined methods are specified in methods.py to make scenario cleaner, 
# except `input_method`, since it requieres an access to global variables
MAS_CONFIG = { # see MAS_DEFAULT_CONFIG in utils.py 
    'verbose': args.verbose, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': args.performance, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'state_dict': MAS_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'redispatch_method': compute_instructions, # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers
    'execute_method': execute_instructions,    # executes the received instructions internally
}

def main():
    """Compose the mosaik scenario and run the simulation."""
    global cells, profiles, net
    set_random_seed(seed=args.seed)
    world = mosaik.World(SIM_CONFIG)

    csv_sim_writer = world.start('CSV_writer', start_date = START_DATE,
                                           output_file=os.path.join(base_dir, args.output_file))
    csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

    gridsim = world.start('Grid', step_size=STEP_SIZE)

    masim = world.start('MAS', step_size=STEP_SIZE, **MAS_CONFIG)

    mosaik_agent = masim.MosaikAgents() # core agent for the mosaik communication 

    pvsim = world.start(
                    "PVSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    
    flsim = world.start(
                    "FLSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    
    wsim = world.start('WecsSim', 
                       step_size=STEP_SIZE, 
                       wind_file=WIND_FILE)

    grid = gridsim.Grid(json=GRID_FILE) #2
    #grid = gridsim.Grid(gridfile=GRID_FILE)

    cells = get_cells_data(grid, gridsim.get_extra_info(), profiles)#

    input_sim = world.start("InputSim", step_size=STEP_SIZE)

    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid', 'Ext_grid']]
    world.connect(ext_grids[0], mosaik_agent, ('P[MW]', 'current'))

    world.connect(mosaik_agent, csv_writer, 'steptime')

    pv = [] # PV simulators
    wp = [] # Wind power simulators
    fl = [] # Flexible Load simulators
    agents = [] # one simple agent per unit (sgen, load)
    cell_controllers = [] # top-level agents that are named as cell_controllers, one per cell
    hierarchical_controllers = []
    for i in [i for i in cells.keys() if 'match' not in i]:
        cell_controllers += masim.MosaikAgents.create(num=1, controller=None)
        entities = list(cells[i]['StaticGen'].values()) + list(cells[i]['Load'].values())
        random.shuffle(entities)
        hierarchical = [cell_controllers[-1]]
        for subset in mit.divide(hierarchy, entities):
            hierarchical += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
            for e in subset:
                    agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                    # 'type-index-bus-cell'
                    if e['type'] == 'StaticGen':
                        if e['bus'] == '7': # wind
                            wp += wsim.WECS.create(num=1, **WECS_CONFIG)
                            e.update({'agent' : agents[-1], 'sim' : wp[-1]})   
                            world.connect(e['sim'], e['agent'], ('P', 'current'))
                            world.connect(e['sim'], csv_writer, 'P') 
                            #print(wp)                    
                        else: # PV
                            pv += pvsim.PVSim.create(num=1, **PVMODEL_PARAMS)
                            e.update({'agent' : agents[-1], 'sim' : pv[-1]})
                            world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                            world.connect(e['agent'], e['sim'], 'scale_factor', weak=True, initial_data={'scale_factor' : 1})
                    elif e['type'] == 'Load':
                        fl += flsim.FLSim.create(num=1)
                        fli = input_sim.Function.create(1, function=lambda x: random.uniform(1, 10)*len(fl)/10)
                        e.update({'agent' : agents[-1], 'sim' : fl[-1]})
                        world.connect(fli[0], e['sim'], ('value', 'P[MW]'))
                        world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                        world.connect(e['agent'], e['sim'], 'scale_factor', weak=True, initial_data={'scale_factor' : 1})
                    else:
                        pass
                    
                    cells.setdefault('match_unit', {})
                    if 'sim' in e:
                        cells['match_unit'].update({e['sim'].eid : e['unit'].eid})

                    #world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                    #world.connect(e['sim'], e['unit'], ('P[MW]', 'P[MW]'))
                    #world.connect(e['agent'], e['sim'], 'scale_factor', weak=True, initial_data={'scale_factor' : 1})
                    
                    if 'unit' in e and 'agent' in e:
                        world.connect(e['unit'], csv_writer, 'P[MW]')
                        world.connect(e['agent'], csv_writer, 'current')
                        world.connect(e['agent'], csv_writer, 'scale_factor')

            hierarchical_controllers += hierarchical[1:]
        

                    
    print('cell_controllers:', len(cell_controllers))
    print('hierarchical_controllers:', len(hierarchical_controllers))
    print('agents:', len(agents))
    #sys.exit()

    check_file_descriptors()
    print(f"Simulation started at {t.ctime()}")
    world.run(until=END)
    print(f"Simulation finished at {t.ctime()}")

if __name__ == '__main__':
    main()
