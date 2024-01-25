"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python example.py

Since neither the simulator in ``mosaik_components/wecssim`` nor the MAS in ``mosaik_components/mas`` are
installed correctly, we add the ``mosaik_components/`` directory to the PYTHONPATH so that
Python will find these modules.

"""
import sys
import os
import json
#from os.path import abspath
#from pathlib import Path
# Add the "src/" dir to the PYTHONPATH to make its packages available for import:
#MODULES_DIR = Path(abspath(__file__)).parent / 'mosaik_components/'
#print(MODULES_DIR)
#sys.path.insert(0, MODULES_DIR)

from pprint import pprint
import pandapower as pp
import pandapower.networks as pn

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

import mosaik
from _mosaik_components.mas.mosaik_agents import *
#from _mosaik_components.mas.utils import *

from cells import create_cells, generate_profiles
from methods import *

base_dir = './'
cells_count = 2

# loading network and flexibility profiles
net_file = os.path.join(base_dir, 'cells.json')
if not os.path.exists(net_file):
    net, net_file = create_cells(cells_count=cells_count)
else:
    net = pp.from_json(net_file)

prof_file = os.path.join(base_dir, 'profiles.json')
if not os.path.exists(prof_file):
    profiles, prof_file = generate_profiles(net)
else:
    with open(prof_file, 'r') as f:
        profiles = json.load(f)

#sys.exit()
SIM_CONFIG = {
    'Grid': {
         #'python': 'mosaik_components.pandapower:Simulator' #2
         'python': 'mosaik_pandapower.simulator:Pandapower'
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
}

END = 3600 * 1# 24 * 1  # 1 day
#START_DATE = '2014-01-01T00:00:00+01:00'
START_DATE = '2014-01-01 12:00:00'
#GRID_FILE = 'demo/pandapower_example.json' 
GRID_FILE = 'demo/cells_net.json' 
WIND_FILE = 'demo/wind_speed_m-s_15min.csv'
STEP_SIZE = 60 * 5

PVSIM_PARAMS = {
    'start_date' : START_DATE,
    'cache_dir' : './', # it caches PVGIS API requests
    'verbose' : False, # print PVGIS parameters and requests
    'gen_neg' : False,
}

PVMODEL_PARAMS = {
    'scale_factor' : 100,
    'lat' : 52.373,
    'lon' : 9.738,
    'optimal_both' : True,
}

#MAS_CONFIG = MAS_DEFAULT_CONFIG.copy()
MAS_CONFIG['verbose'] = 1

WECS_CONFIG = [
    (1, {'P_rated': 5000, 'v_rated': 13, 'v_min': 3.5, 'v_max': 25, 'controller': None}),
]

AGENTS_CONFIG = [
    (8, {}), # PVs
    #(2, {}), # other two
]
CONTROLLERS_CONFIG = [
    (2, {}), # and also two top-level agents that are named as controllers
]

def main():
    """Compose the mosaik scenario and run the simulation."""
    world = mosaik.World(SIM_CONFIG)

    csv_sim_writer = world.start('CSV_writer', start_date = START_DATE,
                                           output_file='results.csv')
    csv_writer = csv_sim_writer.CSVWriter(buff_size = STEP_SIZE)

    csv_sim = world.start('CSV', 
                        sim_start=START_DATE,
                        datafile='fl.csv',
                        delimiter=',')
    csv = csv_sim.FL.create(1)


    gridsim = world.start('Grid', step_size=STEP_SIZE)
    ##pprint(gridsim.meta)
    mas = world.start('MAS', step_size=STEP_SIZE, **MAS_CONFIG)
    #pprint(mas.meta)
    mosaik_agent = mas.MosaikAgents() # core agent for the mosaik communication 

    pv_sim = world.start(
                    "PVSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    #pprint(pv_sim.meta)
    
    fl_sim = world.start(
                    "FLSim",
                    step_size=STEP_SIZE,
                    sim_params=PVSIM_PARAMS,
                )
    
    #wecssim = world.start('WecsSim', step_size=STEP_SIZE, wind_file=WIND_FILE)

    #net = pn.create_cigre_network_mv(with_der="pv_wind")
    #pp.runpp(net, numba=False)
    #print('load', net.res_load)

    #grid = gridsim.Grid(json=GRID_FILE) #2
    grid = gridsim.Grid(gridfile=GRID_FILE)
    #print(grid.children)
    buses = [e for e in grid.children if e.type in ['Bus']]
    gens = [e for e in grid.children if e.type in ['Sgen', 'Gen', 'StaticGen', 'ControlledGen']]
    
    loads = [e for e in grid.children if e.type in ['Load']]
    #print('\n'.join([e.eid for e in loads]))
    #print(grid.res_load)

    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid', 'Ext_grid']]
    world.connect(ext_grids[0], mosaik_agent, ('p_mw', 'current'))

    cell_loads = {int(e.eid.split('Load R')[1]) : e for e in loads if 'Load R' in e.eid}

    pv_gens = {int(e.eid.split('PV ')[1]) : e for e in gens if 'PV' in e.eid}
    pv_loads = {k : v for k, v in cell_loads.items() if k in pv_gens}
    pv_models = {list(pv_gens.keys())[idx] : e for idx, e in enumerate(pv_sim.PVSim.create(len(pv_gens), **PVMODEL_PARAMS))}
    fl_models = {list(pv_loads.keys())[idx] : e for idx, e in enumerate(fl_sim.FLSim.create(len(pv_loads)))}

    #print(fl_models)
    #print(pv_gens)
    #print(pv_loads)
    #print(loads)
    #sys.exit()
    controllers = []
    for n, params in CONTROLLERS_CONFIG:  # iterate over the config sets
        controllers += mas.MosaikAgents.create(num=n, **params)
    # print('controllers:', controllers)


    params.update({'controller' : controllers[0].eid})
    pv_gens_agents = {list(pv_gens.keys())[idx] : e for idx, e in enumerate(mas.MosaikAgents.create(num=len(pv_gens)))}
    pv_loads_agents = {list(pv_loads.keys())[idx] : e for idx, e in enumerate(mas.MosaikAgents.create(num=len(pv_loads)))}

    for idx in pv_gens.keys():
        world.connect(pv_models[idx], pv_gens[idx], ('P[MW]', 'p_mw'))
        world.connect(pv_models[idx], pv_gens_agents[idx], ('P[MW]', 'current')) 
        world.connect(pv_gens_agents[idx], pv_models[idx], 'scale_factor', time_shifted=True, initial_data={'scale_factor' : 1})

        world.connect(pv_models[3], csv_writer, 'P[MW]')
        #world.connect(pv_gens[idx], csv_writer, 'p_mw')
        #world.connect(pv_gens_agents[idx], csv_writer, 'current')
        world.connect(pv_gens_agents[idx], csv_writer, 'scale_factor')
    
    for idx in pv_loads.keys():
        world.connect(csv[0], fl_models[idx], ('Load', 'P[MW]'))
        world.connect(fl_models[idx], pv_loads[idx], ('P[MW]', 'p_mw'))
        world.connect(fl_models[idx], pv_loads_agents[idx], ('P[MW]', 'current')) 
        world.connect(pv_loads_agents[idx], fl_models[idx], 'scale_factor', time_shifted=True, initial_data={'scale_factor' : 1})

        world.connect(fl_models[idx], csv_writer, 'P[MW]')
        #world.connect(pv_loads[idx], csv_writer, 'p_mw')
        #world.connect(pv_loads_agents[idx], csv_writer, 'current')
        world.connect(pv_loads_agents[idx], csv_writer, 'scale_factor')
        

    #for idx in list(pv_gens.keys()) :
        #world.connect(pv_models[idx], csv_writer, 'P[MW]'),
    #    world.connect(pv_gens[idx], csv_writer, 'p_mw')
    #    if idx in pv_loads:
    #        world.connect(pv_loads[idx], csv_writer, 'p_mw')
    #    world.connect(pv_agents[idx], csv_writer, 'current')

        #world.connect(pv_models[idx], pv_gens[idx], ('P[MW]', 'p_mw')),
    #    world.connect(pv_gens[idx], pv_agents[idx], ('p_mw', 'current')) 
    #    if idx in pv_loads:
    #        world.connect(pv_loads[idx], pv_agents[idx], ('p_mw', 'current'))
        #world.connect(pv_agents[idx], pv_gens[idx], ('current', 'p_mw'), weak=True, initial_data={'current' : 0})
        #if idx in pv_loads:
        #    world.connect(pv_agents[idx], pv_loads[idx], ('current', 'p_mw'), weak=True, initial_data={'current' : 0})    

        #world.connect(pv_agents[idx], pv_models[idx], ('current', 'P[MW]'), weak=True, initial_data={'current' : 0}))


    #world.connect(wecs[0], agents[0], ('P', 'current'))
    #world.connect(agents[0], wecs[0], ('current', 'P'), weak=True, initial_data={'current' : 0})

    #agents = []
    #for n, params in AGENTS_CONFIG:  
    #    if len(agents) == 0: # connect the first couple of agents to the first controller
    #        params.update({'controller' : controllers[0].eid})
    #    else:
    #        params.update({'controller' : controllers[1].eid})
    #    agents += mas.MosaikAgents.create(num=n, **params)
    # print('agents:', agents)

    #wecs = []
    #for n_wecs, params in WECS_CONFIG:
    #    for _ in range(n_wecs):
    #        w = wecssim.WECS(**params)
    #        wecs.append(w)

    #world.connect(wecs[0], agents[0], ('P', 'current'))
    #world.connect(agents[0], wecs[0], ('current', 'P'), weak=True, initial_data={'current' : 0})

    #world.connect(wecs[0], gens[2], ('P', 'p_mw'))

    
    #world.connect(wecs[0], agents[0], ('P', 'current'))
    #world.connect(agents[0], wecs[0], ('current', 'P'), weak=True, initial_data={'current' : 0})
    #world.connect(wecs[0], gens[2], ('P', 'p_mw'))            


    #world.connect(gens[0], agents[1], ('p_mw', 'current'))
    #world.connect(loads[0], agents[1], ('p_mw', 'current'))
    #world.connect(gens[1], agents[2], ('p_mw', 'current'))
    #world.connect(loads[1], agents[3], ('p_mw', 'current'))
     # connect the external network to the core agent
                                                                    # to execute the default redispatch algorithm
                                                                    # which is based on the core agent state

    
    #world.connect(ext_grids[0], csv_writer, 'P[MW]')

    #world.connect(controllers[0], csv_writer, 'current')
    #world.connect(controllers[1], csv_writer, 'current')    
    #world.connect(agents[0], csv_writer, 'current')
    #world.connect(agents[1], csv_writer, 'current')
    #world.connect(agents[2], csv_writer, 'current')
    #world.connect(agents[3], csv_writer, 'current')

    world.run(END)

if __name__ == '__main__':
    main()
