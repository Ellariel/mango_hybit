"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python scenario.py

"""
import sys
import os
import json
import random
import argparse
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

import mosaik
from _mosaik_components.mas.mosaik_agents import *
from _mosaik_components.mas.utils import set_random_seed

#from cells import create_cells, generate_profiles
from methods import *

parser = argparse.ArgumentParser()
parser.add_argument('--cells', default=2, type=int)
parser.add_argument('--verbose', default=0, type=int)
#parser.add_argument('--clean', default=False, type=bool)
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--output_file', default='results.csv', type=str)
parser.add_argument('--performance', default=True, type=bool)
args = parser.parse_args()

base_dir = args.dir
cells_count = args.cells

# loading network and flexibility profiles
cells = {}

net_file = os.path.join(base_dir, 'cells.json')
#if not os.path.exists(net_file) or args.clean:
#    net, net_file = create_cells(cells_count=cells_count)
#else:
if os.path.exists(net_file):
    net = pp.from_json(net_file)
else:
    print(f"no file error: {net_file}")
    sys.exit()

prof_file = os.path.join(base_dir, 'profiles.json')
#if not os.path.exists(prof_file) or args.clean:
#    profiles, prof_file = generate_profiles(net, seed=args.seed)
#else:
if os.path.exists(prof_file):
    with open(prof_file, 'r') as f:
        profiles = json.load(f)
else:
    print(f"no file error: {prof_file}")
    sys.exit()

END = 3600 * 24 * 1  # 1 day
START_DATE = '2014-01-01 12:00:00'
GRID_FILE = net_file #'demo/cells_net.json' 
WIND_FILE = 'demo/wind_speed_m-s_15min.csv'
STEP_SIZE = 60 * 15

# simulators
SIM_CONFIG = {
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator' #2
         # 'python': 'mosaik_pandapower.simulator:Pandapower'
    },
    'PVSim': {
        'python': '_mosaik_components.pv.pvgis_simulator:PVGISSimulator',
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
    'cache_dir' : './', # it caches PVGIS API requests
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
WECS_CONFIG = [
    (1, {'P_rated': 5000, 'v_rated': 13, 'v_min': 3.5, 'v_max': 25, 'controller': None}),
]

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
                #print('LoadQQQQQQQQQQQQQQ', input_data, profile, value, scale_factor)
            elif 'Gen' in eid or 'PV' in eid:
                state['production']['min'] = profile['min']
                state['production']['max'] = min(abs(value), profile['max'])
                state['production']['current'] = np.clip(abs(value), profile['min'], min(abs(value), profile['max']))
                #print('GenQQQQQQQQQQQQQQ', input_data, profile, value, scale_factor)
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
    world = mosaik.World(SIM_CONFIG)

    csv_sim_writer = world.start('CSV_writer', start_date = START_DATE,
                                           output_file=os.path.join(base_dir, args.output_file))
    csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

    #csv_sim = world.start('CSV', 
    #                    sim_start=START_DATE,
    #                    datafile='fl.csv',
    #                    delimiter=',')
    #csv = csv_sim.FL.create(1)


    gridsim = world.start('Grid', step_size=STEP_SIZE)
    ##pprint(gridsim.meta)
    masim = world.start('MAS', step_size=STEP_SIZE, **MAS_CONFIG)
    #pprint(mas.meta)
    mosaik_agent = masim.MosaikAgents() # core agent for the mosaik communication 

    pvsim = world.start(
                    "PVSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    #pprint(pv_sim.meta)
    
    flsim = world.start(
                    "FLSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    
    #wecssim = world.start('WecsSim', step_size=STEP_SIZE, wind_file=WIND_FILE)

    #net = pn.create_cigre_network_mv(with_der="pv_wind")
    #pp.runpp(net, numba=False)
    #print('load', net.res_load)

    grid = gridsim.Grid(json=GRID_FILE) #2
    #grid = gridsim.Grid(gridfile=GRID_FILE)

    cells = get_cells_data(grid, gridsim.get_extra_info(), profiles)
    #print(profiles)
    #print(cells)

    input_sim = world.start("InputSim", step_size=STEP_SIZE)

    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid', 'Ext_grid']]
    world.connect(ext_grids[0], mosaik_agent, ('P[MW]', 'current'))

    world.connect(mosaik_agent, csv_writer, 'steptime')

    pv = [] # PV simulators
    wp = [] # Wind power simulators
    fl = [] # Flexible Load simulators
    agents = [] # one simple agent per unit (sgen, load)
    controllers = [] # top-level agents that are named as controllers, one per cell
    for i in [i for i in cells.keys() if 'match' not in i]:
        controllers += masim.MosaikAgents.create(num=1, controller=None)
        for k in ['StaticGen', 'Load']:
            if k in cells[i]:
                for e in cells[i][k].values():
                    agents += masim.MosaikAgents.create(num=1, controller=controllers[-1].eid)
                    if k == 'StaticGen':
                        pv += pvsim.PVSim.create(num=1, **PVMODEL_PARAMS)
                        e.update({'agent' : agents[-1], 'sim' : pv[-1]})
                    elif k == 'Load':
                        fl += flsim.FLSim.create(num=1)
                        fli = input_sim.Function.create(1, function=lambda x: random.uniform(1, 10)*len(fl)/10)
                        e.update({'agent' : agents[-1], 'sim' : fl[-1]})
                        world.connect(fli[0], e['sim'], ('value', 'P[MW]'))
                    else:
                        pass
                    
                    cells.setdefault('match_unit', {})
                    cells['match_unit'].update({e['sim'].eid : e['unit'].eid})

                    world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                    #world.connect(e['sim'], e['unit'], ('P[MW]', 'P[MW]'))
                    world.connect(e['agent'], e['sim'], 'scale_factor', weak=True, initial_data={'scale_factor' : 1})
                    
                    #world.connect(e['unit'], csv_writer, 'P[MW]')
                    world.connect(e['unit'], csv_writer, 'P[MW]')
                    world.connect(e['agent'], csv_writer, 'current')
                    world.connect(e['agent'], csv_writer, 'scale_factor')
                    
    #print(controllers)
    #print(agents)        
    #print(cells)
    #sys.exit()
    set_random_seed(seed=args.seed)
    print(f"Simulation started at {t.ctime()}")
    world.run(END)
    print(f"Simulation finished at {t.ctime()}")

if __name__ == '__main__':
    main()
