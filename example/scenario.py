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
import mosaik.util
import random
from pathlib import Path
import argparse
import more_itertools as mit
import pandapower as pp
import nest_asyncio
nest_asyncio.apply()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from cells import *
#from cells import _randomize
from methods import *

parser = argparse.ArgumentParser()
parser.add_argument('--cells', default=3, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--clean', default=True, type=bool)
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--output_file', default='results.csv', type=str)
parser.add_argument('--performance', default=False, type=bool)
parser.add_argument('--hierarchy', default=1, type=int)
args = parser.parse_args()

set_random_seed(seed=args.seed)

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

#print(net.sgen)
#print(net.bus)
#print(_randomize())
#sys.exit()

END = 60*15 * 1#3600 * 24 * 1  # 1 day 
START_DATE = '2014-07-01 12:00:00'
DATE_FORMAT = 'mixed' #'YYYY-MM-DD hh:mm:ss'
STEP_SIZE = 60 * 15
GRID_FILE = net_file
WIND_FILE = os.path.join(data_dir, 'wind_speed_m-s_15min.csv')
LOAD_FILE = os.path.join(data_dir, 'syntetic_data_15min.csv')


prof_file = os.path.join(data_dir, 'profiles.json')
if not os.path.exists(prof_file) or args.clean:
    profiles, prof_file = generate_profiles(net, seed=args.seed, dir=data_dir, start=START_DATE, end=END, step_size=STEP_SIZE)
else:
    with open(prof_file, 'r') as f:
        profiles = json.load(f)

data = None
for k, v in profiles.items():
    if 'current' in v:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame().from_dict(v['current'])
        else:
            data = pd.concat([data, pd.DataFrame().from_dict(v['current'])[[k]]], axis=1)
data.to_csv(LOAD_FILE, index=False)
data = Path(LOAD_FILE)
data.write_text(f"Profiles\n{data.read_text()}")

# simulators
SIM_CONFIG = {
    'PVSim': {
        'python': 'simulators.pv:Simulator',
    },
    'FLSim': {
        'python': 'simulators.flexible:Simulator',
    },
    'FGSim': {
        'python': 'simulators.flexible:Simulator',
    },
    'WPSim': {
        'python': 'simulators.wecssim:Simulator',
    },
    'GridSim': {
         'python': 'mosaik_components.pandapower:Simulator',
    },
    'MAS': {
        'python': 'mosaik_components.mas:MosaikAgents',
    },
    'OutputSim': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    'InputSim': {
        'python': 'mosaik_csv:CSV'
    },  
}

# The simulator meta data that we return in "init()":
MAS_META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'MosaikAgents': {
            'public': True,
            'any_inputs': True,
            'params': ['controller', 'initial_state'],
            'attrs': ['current', 'scale_factor', 'steptime', 'production_delta[MW]', 'consumption_delta[MW]', 'consumption[MW]', 'production[MW]'],
        },
    },
}

# PV simulator
PVSIM_PARAMS = {
    'start_date' : START_DATE,
    'cache_dir' : data_dir, # it caches PVGIS API requests
    'verbose' : False, # print PVGIS parameters and requests
    'gen_neg' : False, # return negative P
}

# For each PV
PVMODEL_PARAMS = {
    'scale_factor' : 10000,
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
    #print(input_data)
    #def update_flexibility(a, b):
    #    a = a.copy()
    #    a['max'] = b['max']
    #    a['min'] = b['min']
    #    #if 'current' in a and 'current' in b and isinstance(b['current'], (int, float, list)):
    #    #    a['current'] = b['current']
    #    #if 'scale_factor' in a and 'scale_factor' in b and isinstance(b['scale_factor'], (int, float, list)):
    #    #    a['scale_factor'] = b['scale_factor']
    #    return a

    global cells
    state = copy.deepcopy(MAS_STATE)
    profile = get_unit_profile(aeid, cells)
    #print(aeid, input_data)
    for eid, value in input_data.items():
        value = list(value.values())[0]
        if 'Load' in eid:
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption'] = update_flexibility(state['consumption'], profile)
                state['consumption']['current'] = value
                #print('Load', state)
        elif 'PV' in eid or 'WECS' in eid:
                if first_time_step:
                    profile['max'] = min(abs(value), profile['max'])
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = update_flexibility(state['production'], profile)
                state['production']['current'] = value
                #print('Gen', state)
        elif 'StaticGen' in eid:
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = update_flexibility(state['production'], profile)
                state['production']['current'] = value
                #print('Gen', state)
        elif 'ExternalGrid' in eid:
                state['production'] = update_flexibility(state['production'], profile)
                state['consumption'] = update_flexibility(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['current'] = value
                else:
                    state['consumption']['current'] = abs(value) 
        break 

    state['consumption']['scale_factor'] = current_state['consumption']['scale_factor']
    state['production']['scale_factor'] = current_state['production']['scale_factor']
    return state
    '''
    if 'production[MW]' in input_data:
        for eid, value in input_data['current'].items():
            if 'Gen' in eid or 'PV' in eid or 'Wecs' in eid: #in eid or 'PV' in eid or 'Wecs'
                if first_time_step:
                    profile['max'] = min(abs(value), profile['max'])
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = _update(state['production'], profile)
                state['production']['current'] += value
            elif 'ExternalGrid' in eid:
                state['production'] = _update(state['production'], profile)
                state['consumption'] = _update(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['current'] += value
                else:
                    state['consumption']['current'] += abs(value) 



    for attr in input_data:
        if attr == 'production[MW]':
            for eid, value in input_data['current'].items():

            data.update({attr : current_state['production']['current']})
            if 'production_delta[MW]' in attrs and not converged:
                data.update({'production_delta[MW]' : current_state['production']['scale_factor']})
        elif attr == 'consumption[MW]':
            data.update({attr : current_state['consumption']['current']})
            if 'consumption_delta[MW]' in attrs and not converged:
                data.update({'consumption_delta[MW]' : current_state['consumption']['scale_factor']})



    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            if 'Load' in eid or 'FL' in eid: # check the type of connected unit and its profile in eid or 'FL' 
                state['consumption'] = _update(state['consumption'], profile)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption']['current'] += value
                #print('Load', state, profile)
                #print('input0','Load', profile, state)
            elif 'Gen' in eid or 'PV' in eid or 'Wecs' in eid: #in eid or 'PV' in eid or 'Wecs'
                if first_time_step:
                    profile['max'] = min(abs(value), profile['max'])
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = _update(state['production'], profile)
                state['production']['current'] += value
            elif 'ExternalGrid' in eid:
                state['production'] = _update(state['production'], profile)
                state['consumption'] = _update(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['current'] += value
                else:
                    state['consumption']['current'] += abs(value) 
            break 
    else:
        for eid, value in input_data.items():
            value = list(value.values())[0]
            if 'Load' in eid: # check the type of connected unit and its profile in eid or 'FL' 
                #print('input', eid, value,'Load', profile, state)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption'] = _update(state['consumption'], profile)
                state['consumption']['current'] += value
                #print('Load', state)
            elif 'Gen' in eid: #in eid or 'PV' in eid or 'Wecs'
                if first_time_step:
                    profile['max'] = min(abs(value), profile['max'])
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = _update(state['production'], profile)
                state['production']['current'] += value
                #print('Gen', state)
            elif 'ExternalGrid' in eid:
                state['production'] = _update(state['production'], profile)
                state['consumption'] = _update(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['current'] += value
                else:
                    state['consumption']['current'] += abs(value) 
            break 

    state['consumption']['scale_factor'] = current_state['consumption']['scale_factor']
    state['production']['scale_factor'] = current_state['production']['scale_factor']
    return state
    '''

# Multi-agent system (MAS) configuration
# User-defined methods are specified in methods.py to make scenario cleaner, 
# except `input_method`, since it requieres an access to global variables
MAS_CONFIG = { # see MAS_DEFAULT_CONFIG in utils.py 
    'META': MAS_META,
    'verbose': args.verbose, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': args.performance, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'convergence_steps' : 1, # higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence
    'state_dict': MAS_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'execute_method': execute_instructions,    # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers,
                                               # executes the received instructions internally
    'initialize' : initialize,
    'finalize' : finalize,
}

def main():
    """Compose the mosaik scenario and run the simulation."""
    global cells, profiles, net
    world = mosaik.World(SIM_CONFIG)
    gridsim = world.start('GridSim', step_size=STEP_SIZE)
    grid = gridsim.Grid(json=GRID_FILE)

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
        #flsim = world.start(
        #                "FLSim",
        #                step_size=STEP_SIZE,
        #                csv_file=LOAD_FILE,
        #                sim_params=dict(gen_neg=True),
        #            )
        wsim = world.start('WPSim', 
                        step_size=STEP_SIZE, 
                        wind_file=WIND_FILE,
                    )
        isim = world.start("InputSim", 
                              sim_start=START_DATE, 
                              date_format=DATE_FORMAT,
                              datafile=LOAD_FILE)
        
    output_sim = world.start('OutputSim', start_date = START_DATE,
                                            output_file=os.path.join(results_dir, args.output_file))
    report = output_sim.CSVWriter(buff_size=STEP_SIZE)
    inputs = isim.Profiles.create(1)[0]
    mosaik_agent = masim.MosaikAgents() # core agent for the mosaik communication 

    cells = get_cells_data(grid, gridsim.get_extra_info(), profiles)
    cells.setdefault('match_unit', {})
    cells.setdefault('match_agent', {})
    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid', 'Ext_grid']]
    buses = [e for e in grid.children if e.type in ['Bus']]
    cells['match_agent'].update({'MosaikAgent' : ext_grids[0].eid})
    world.connect(ext_grids[0], mosaik_agent, ('P[MW]', 'ExternalGrid'), time_shifted=True, initial_data={'P[MW]': 0})
    world.connect(mosaik_agent, report, 'steptime')
    world.connect(mosaik_agent, report, 'production[MW]')
    world.connect(mosaik_agent, report, 'consumption[MW]')
    world.connect(ext_grids[0], report, 'P[MW]')

    pv = [] # PV simulators
    wp = [] # Wind power simulators
    #fl = [] # Flexible Load simulators
    agents = [] # one simple agent per unit (sgen, load)
    cell_controllers = [] # top-level agents that are named as cell_controllers, one per cell
    hierarchical_controllers = []

    connected_loads = 0
    connected_gens = 0

    for i in [i for i in cells.keys() if 'match' not in i]:
        cell_controllers += masim.MosaikAgents.create(num=1, controller=None)
        #world.connect(cell_controllers[-1], report, 'current')

        entities = list(cells[i]['StaticGen'].values()) + list(cells[i]['Load'].values())
        random.shuffle(entities)
        hierarchical = [cell_controllers[-1]]
        for subset in mit.divide(hierarchy, entities):
            hierarchical += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
            for e in subset:
                    
                    # 'type-index-bus-cell'
                    bus_eid = '-'.join(e['bus'].split('-', )[:-1])
                    bus = [i for i in buses if i.eid == bus_eid][0]
                    if e['type'] == 'StaticGen':
                        if '-7' in e['bus']: # wind at Bus-*-7    
                            agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)                 
                            wp += wsim.WECS.create(num=1, **WECS_CONFIG)
                            e.update({'agent' : agents[-1], 'sim' : wp[-1]}) 
                            world.connect(e['sim'], bus, ('P[MW]', 'P_gen[MW]'))
                            #world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                            world.connect(e['sim'], e['agent'], ('P[MW]', e['sim'].eid))   
                            #world.connect(e['agent'], e['sim'], 'scale_factor', weak=True)
                            world.connect(e['agent'], e['sim'], ('production_delta[MW]', 'scale_factor'), weak=True)
                            world.connect(e['sim'], report, 'P[MW]') 
                            #world.connect(e['agent'], report, 'production[MW]') 
                        elif '-3' in e['bus']:
                            agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                            pv += pvsim.PVSim.create(num=1, **PVMODEL_PARAMS)
                            e.update({'agent' : agents[-1], 'sim' : pv[-1]})   
                            world.connect(e['sim'], bus, ('P[MW]', 'P_gen[MW]'))
                            #world.connect(e['sim'], e['agent'], ('P[MW]', 'current'))
                            world.connect(e['sim'], e['agent'], ('P[MW]', e['sim'].eid))
                            #world.connect(e['agent'], e['sim'], 'scale_factor', weak=True)
                            world.connect(e['agent'], e['sim'], ('production_delta[MW]', 'scale_factor'), weak=True)
                            world.connect(e['sim'], report, 'P[MW]') 
                            #world.connect(e['agent'], report, 'production[MW]') 
                        else:
                            agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                            e.update({'agent' : agents[-1], 'sim' : inputs})   
                            world.connect(e['sim'], bus, (e['name'], 'P_gen[MW]'))
                            world.connect(e['sim'], e['agent'], e['name'])#(e['name'], 'current'))
                            world.connect(e['sim'], report, e['name'])#(e['name'], 'P[MW]'))
                        connected_gens += 1
                            #print(e)
                    elif e['type'] == 'Load':
                            agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                            e.update({'agent' : agents[-1], 'sim' : inputs})   
                            world.connect(e['sim'], bus, (e['name'], 'P_load[MW]'))
                            world.connect(e['sim'], e['agent'], e['name'])#(e['name'], 'current'))
                            world.connect(e['sim'], report, e['name'])#(e['name'], 'P[MW]'))
                            connected_loads += 1

                            



                    #    if '-3' in e['bus']:
                    #        agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                    #        fl += flsim.FLSim.create(num=1)
                    #        e.update({'agent' : agents[-1], 'sim' : fl[-1]})
                    #        world.connect(e['sim'], bus, ('P[MW]', 'P_load[MW]'))
                        #else:
                        #    world.connect(bus, report, 'P[MW]') 
                    #else:
                    #    pass
                        
                    
                    if 'sim' in e:
                        
                        
                        
                        #world.connect(e['agent'], report, 'current')
                        #world.connect(e['agent'], report, 'scale_factor')
                        if e['sim'] == inputs:
                            cells['match_unit'].update({e['name'] : e['unit'].eid})
                            cells['match_agent'].update({e['agent'].eid : e['name']})
                        else:
                            cells['match_unit'].update({e['sim'].eid : e['unit'].eid})
                            cells['match_agent'].update({e['agent'].eid : e['sim'].eid})
                        

            hierarchical_controllers += hierarchical[1:]
    #print(cells['match_unit'])
    #print(cells['match_agent'])
    #sys.exit()
    print('connected_loads', connected_loads)
    print('connected_gens', connected_gens)
    gridsim.disable_elements(cells['match_unit'].values())

    print('cell controllers:', len(cell_controllers))
    print('hierarchical controllers:', len(hierarchical_controllers))
    print('power unit agents:', len(agents))
    if args.performance:
        mosaik.util.plot_dataflow_graph(world, hdf5path=os.path.join(results_dir, '.hdf5'), show_plot=False)
    print(f"Simulation started at {time.ctime()}")
    world.run(until=END, print_progress='individual' if args.performance else True)
    #world.run(until=END, print_progress=True)
    print(f"Simulation finished at {time.ctime()}")

if __name__ == '__main__':
    main()