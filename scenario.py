"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python example.py

Since neither the simulator in ``mosaik_components/wecssim`` nor the MAS in ``mosaik_components/mas`` are
installed correctly, we add the ``mosaik_components/`` directory to the PYTHONPATH so that
Python will find these modules.

"""
import sys
from os.path import abspath
from pathlib import Path
# Add the "src/" dir to the PYTHONPATH to make its packages available for import:
MODULES_DIR = Path(abspath(__file__)).parent / 'mosaik_components/'
print(MODULES_DIR)
sys.path.insert(0, MODULES_DIR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

import mosaik
from mosaik_components.mas.mosaik_agents import *
from mosaik_components.mas.utils import *

SIM_CONFIG = {
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator'
    },
    'WecsSim': {
        'python': 'mosaik_components.wecssim.mosaik:WecsSim',
    },
    'MAS': {
        'python': 'mosaik_components.mas.mosaik_agents:MosaikAgents'
    },
    'CSV_writer': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
}

MAS_CONFIG = MAS_DEFAULT_CONFIG.copy()
MAS_CONFIG['verbose'] = 0

END = 3600 * 6# 24 * 1  # 1 day
START_DATE = '2014-01-01T00:00:00+01:00'
START_DATE = '2014-01-01 00:00:00'
#GRID_FILE = 'demo/pandapower_example.json' 
GRID_FILE = 'demo/cells_example.json' 
WIND_FILE = 'demo/wind_speed_m-s_15min.csv'
STEP_SIZE = 60 * 5

'''
MAS_DEFAULT_CONFIG = {
    'verbose': 1, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'state_dict': MAS_DEFAULT_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'redispatch_method': compute_instructions, # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers
}
'''

WECS_CONFIG = [
    (1, {'P_rated': 5000, 'v_rated': 13, 'v_min': 3.5, 'v_max': 25, 'controller': None}),
]

AGENTS_CONFIG = [
    (2, {}), # here we configure two agents with empty parameters
    (2, {}), # other two
]
CONTROLLERS_CONFIG = [
    (2, {}), # and also two top-level agents that are named as controllers
]

def main():
    """Compose the mosaik scenario and run the simulation."""
    world = mosaik.World(SIM_CONFIG)
    wecssim = world.start('WecsSim', step_size=STEP_SIZE, wind_file=WIND_FILE)
    gridsim = world.start('Grid', step_size=STEP_SIZE)
    mas = world.start('MAS', step_size=STEP_SIZE, **MAS_CONFIG)

    csv_sim_writer = world.start('CSV_writer', start_date = START_DATE,
                                           output_file='results.csv')
    csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

    grid = gridsim.Grid(json=GRID_FILE)
    # print(grid.children)
    buses = [e for e in grid.children if e.type in ['Bus']]
    gens = [e for e in grid.children if e.type in ['Gen', 'StaticGen', 'ControlledGen']]
    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid']]
    loads = [e for e in grid.children if e.type in ['Load']]

    mosaik_agent = mas.MosaikAgents() # core agent for the mosaik communication 

    controllers = []
    for n, params in CONTROLLERS_CONFIG:  # iterate over the config sets
        controllers += mas.MosaikAgents.create(num=n, **params)
    # print('controllers:', controllers)

    agents = []
    for n, params in AGENTS_CONFIG:  
        if len(agents) == 0: # connect the first couple of agents to the first controller
            params.update({'controller' : controllers[0].eid})
        else:
            params.update({'controller' : controllers[1].eid})
        agents += mas.MosaikAgents.create(num=n, **params)
    # print('agents:', agents)

    wecs = []
    for n_wecs, params in WECS_CONFIG:
        for _ in range(n_wecs):
            w = wecssim.WECS(**params)
            wecs.append(w)

    world.connect(wecs[0], agents[0], ('P', 'current'))
    world.connect(agents[0], wecs[0], ('current', 'P'), weak=True, initial_data={'current' : 0})

    world.connect(gens[0], agents[1], ('P[MW]', 'current'))
    world.connect(loads[0], agents[1], ('P[MW]', 'current'))
    world.connect(gens[1], agents[2], ('P[MW]', 'current'))
    world.connect(loads[1], agents[3], ('P[MW]', 'current'))
    world.connect(ext_grids[0], mosaik_agent, ('P[MW]', 'current')) # connect the external network to the core agent
                                                                    # to execute the default redispatch algorithm
                                                                    # which is based on the core agent state

    
    #world.connect(ext_grids[0], csv_writer, 'P[MW]')

    world.connect(controllers[0], csv_writer, 'current')
    world.connect(controllers[1], csv_writer, 'current')    
    world.connect(agents[0], csv_writer, 'current')
    world.connect(agents[1], csv_writer, 'current')
    world.connect(agents[2], csv_writer, 'current')
    world.connect(agents[3], csv_writer, 'current')

    world.run(END)

if __name__ == '__main__':
    main()
