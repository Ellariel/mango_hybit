"""
This file contains the mosaik scenario. To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import os
import sys
import time
import json
import mosaik
import random
import argparse
import numpy as np
import more_itertools as mit
import pandapower as pp

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

from cells import *
from methods import *

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
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
cells_count = args.cells
hierarchy = args.hierarchy

print(f"cells count: {cells_count}")
print(f"hierarchy depth: {hierarchy}")

# loading network and flexibility profiles
net_file = os.path.join(data_dir, 'cells.json')
if not os.path.exists(net_file) or args.clean:
    net, net_file = create_cells(cells_count=cells_count, dir=data_dir)
else:
    net = pp.from_json(net_file)
pp.runpp(net, numba=False)
print(f"loads: {len(net.load)}, gens: {len(net.sgen)}")

prof_file = os.path.join(data_dir, 'profiles.json')
if not os.path.exists(prof_file) or args.clean:
    profiles, prof_file = generate_profiles(net, seed=args.seed, dir=data_dir)
else:
    with open(prof_file, 'r') as f:
        profiles = json.load(f)

END = 3600 * 24 * 1  # 1 day
START_DATE = '2014-01-01 08:00:00'
DATE_FORMAT = 'YYYY-MM-DD hh:mm:ss'
GRID_FILE = net_file
WIND_FILE = os.path.join(data_dir, 'wind_speed_m-s_15min.csv')
LOAD_FILE = os.path.join(data_dir, 'Braunschweig_meteodata_2020_15min.csv')
STEP_SIZE = 60 * 15

# simulators
SIM_CONFIG = {
    'PVSim': {
        'python': 'simulators.pv.pvgis_simulator:PVGISSimulator',
    },
    'FLSim': {
        'python': 'simulators.flexible.flexiload_simulator:FLSimulator',
    },
    'WPSim': {
        'python': 'simulators.wecssim.wind_simulator:WecsSim',
    },
    'MAS': {
        'python': '_mosaik_components.mas.mosaik_agents:MosaikAgents',
    },
    'GridSim': {
         'python': 'mosaik_components.pandapower:Simulator',
    },
    'OutputSim': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    'InputSim': {
        'python': 'mosaik.basic_simulators.input_simulator:InputSimulator',
    },
    'LoadSim': {
        'python': 'mosaik_csv:CSV'
    },  
}

# PV simulator
PVSIM_PARAMS = {
    'start_date' : START_DATE,
    'cache_dir' : os.path.join(base_dir, 'data/'), # it caches PVGIS API requests
    'verbose' : False, # print PVGIS parameters and requests
    'gen_neg' : False, # return negative P
}

# For each PV
PVMODEL_PARAMS = {
    'scale_factor' : 1000000,
    'lat' : 52.373,
    'lon' : 9.738,
    'optimal_both' : True,
}

# Wind power
WECS_CONFIG = {'P_rated': 7000, 
               'v_rated': 13, 
               'v_min': 3.5, 
               'v_max': 25}

def input_to_state(aeid, aid, input_data, current_state, current_time, first_time_step, **kwargs):
    # aeid: Agent_3
    # aid: agent3
    # input_data: {'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0, 'FLSim-0.FLSim-0': 0.9}}
    # current_state: {'production': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}, 
    #                'consumption': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}}
    global cells
    state = copy.deepcopy(MAS_STATE)
    profile = get_unit_profile(aeid, cells)
    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            if 'Load' in eid or 'FL' in eid: # check the type of connected unit and its profile
                state['consumption']['current'] += abs(value)
                state['consumption'].update(profile)
            elif 'Gen' in eid or 'PV' in eid or 'Wecs' in eid:
                state['production']['current'] += abs(value)
                if first_time_step:
                    profile['max'] = min(abs(value), profile['max'])
                state['production'].update(profile)
            elif 'ExternalGrid' in eid:
                if value > 0: # check the convention here!
                    state['production']['current'] += value
                else:
                    state['consumption']['current'] += abs(value) 
                state['production'].update(profile)
                state['consumption'].update(profile) 
            break 
    state['consumption']['scale_factor'] = current_state['consumption']['scale_factor']
    state['production']['scale_factor'] = current_state['production']['scale_factor']
    return state

# Multi-agent system (MAS) configuration
# User-defined methods are specified in methods.py to make scenario cleaner, 
# except `input_method`, since it requieres an access to global variables
MAS_CONFIG = { # see MAS_DEFAULT_CONFIG in utils.py 
    'verbose': args.verbose, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': args.performance, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'convergence_steps' : 3, # higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence
    'state_dict': MAS_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'execute_method': execute_instructions,    # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers,
                                               # executes the received instructions internally
}

def main():
    """Compose the mosaik scenario and run the simulation."""
    global cells, profiles, net
    set_random_seed(seed=args.seed)
    world = mosaik.World(SIM_CONFIG)

    gridsim = world.start('GridSim', step_size=STEP_SIZE)
    grid = gridsim.Grid(json=GRID_FILE)

    #loadsim = world.start("LoadSim", 
    #                      sim_start=START_DATE, 
    ##                      date_format=DATE_FORMAT,
     #                     datafile=LOAD_FILE)

    with world.group():
        masim = world.start('MAS', 
                            step_size=STEP_SIZE, 
                            **MAS_CONFIG,
                            )
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
        wsim = world.start('WPSim', 
                        step_size=STEP_SIZE, 
                        wind_file=WIND_FILE,
                    )
    input_sim = world.start("InputSim", step_size=STEP_SIZE)   
    output_sim = world.start('OutputSim', start_date = START_DATE,
                                            output_file=os.path.join(results_dir, args.output_file))
    report = output_sim.CSVWriter(buff_size=STEP_SIZE)
    mosaik_agent = masim.MosaikAgents() # core agent for the mosaik communication 

    cells = get_cells_data(grid, gridsim.get_extra_info(), profiles)
    cells.setdefault('match_unit', {})
    cells.setdefault('match_agent', {})
    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid', 'Ext_grid']]
    cells['match_agent'].update({'MosaikAgent' : ext_grids[0].eid})
    world.connect(ext_grids[0], mosaik_agent, ('P[MW]', 'current'))
    world.connect(mosaik_agent, report, 'steptime')
    world.connect(mosaik_agent, report, 'current')

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
                        if '-7' in e['bus']: # wind at Bus-*-7                     
                            wp += wsim.WECS.create(num=1, **WECS_CONFIG)
                            e.update({'agent' : agents[-1], 'sim' : wp[-1]})         
                        else: # PV
                            pv += pvsim.PVSim.create(num=1, **PVMODEL_PARAMS)
                            e.update({'agent' : agents[-1], 'sim' : pv[-1]})             
                    elif e['type'] == 'Load':
                        fl += flsim.FLSim.create(num=1)
                        l = input_sim.Function.create(1, function=lambda x: x * len(fl)/1000)
                        #fli = loadsim.Braunschweig.create(1)
                        #world.connect(l[0], csv_writer, ('value', 'P[MW]'))
                        e.update({'agent' : agents[-1], 'sim' : fl[-1]})
                        world.connect(l[0], e['sim'], ('value', 'P[MW]'))
                    else:
                        pass
                    
                    if 'sim' in e:
                        cells['match_unit'].update({e['sim'].eid : e['unit'].eid})
                        if 'agent' in e:
                            cells['match_agent'].update({e['agent'].eid : e['sim'].eid})
                            world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                            world.connect(e['agent'], e['sim'], 'scale_factor', weak=True)
                            #world.connect(e['sim'], e['unit'], 'P[MW]')
                            world.connect(e['sim'], report, 'P[MW]') 
                            world.connect(e['agent'], report, 'current')
                            world.connect(e['agent'], report, 'scale_factor')
                    #break
            hierarchical_controllers += hierarchical[1:]
    
    print('cell controllers:', len(cell_controllers))
    print('hierarchical controllers:', len(hierarchical_controllers))
    print('power unit agents:', len(agents))

    print(f"Simulation started at {time.ctime()}")
    world.run(until=END)
    print(f"Simulation finished at {time.ctime()}")

if __name__ == '__main__':
    main()
