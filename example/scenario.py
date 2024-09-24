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
#from pathlib import Path
import argparse
#import more_itertools as mit
import pandapower as pp
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#from cells import *
from methods import *
from mosaik_components.mas.mosaik_agents import META

import nest_asyncio
nest_asyncio.apply()

parser = argparse.ArgumentParser()
parser.add_argument('--cells', default=2, type=int)
parser.add_argument('--verbose', default=0, type=int)
#parser.add_argument('--clean', default=True, type=bool)
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--output_file', default='results.csv', type=str)
parser.add_argument('--performance', default=True, type=bool)
#parser.add_argument('--hierarchy', default=1, type=int)
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
END = 60*60 + 1 # 1 day
STEP_SIZE = 60*15

SIM_CONFIG = {
    #'PVSim': {
    #    'python': 'simulators.pv:Simulator',
    #},
    'FlexSim': {
        'python': 'simulators.flexible:Simulator',
    },
    #'FGSim': {
    #    'python': 'simulators.flexible:Simulator',
    #},
    #'WPSim': {
    #    'python': 'simulators.wecssim:Simulator',
    #},
    'GridSim': {
         'python': 'mosaik_components.pandapower:Simulator',
    },
    'MASSCA': {
        'python': 'mosaik_components.mas:MosaikAgents',
    },
    'OutputSim': {
        'python': 'mosaik_csv_writer:CSVWriter',
    },
    'InputSim': {
        'python': 'mosaik_csv:CSV'
    },  
}




#sys.exit()

# The simulator meta data that we return in "init()":
MAS_META = META.copy()
MAS_META['models']['MosaikAgents']['attrs'] += ['production_delta[MW]', 
                                                'consumption_delta[MW]', 
                                                'consumption[MW]', 
                                                'production[MW]']

# PV simulator
#PVSIM_PARAMS = {
#    'start_date' : START_DATE,
#    'cache_dir' : data_dir, # it caches PVGIS API requests
#    'verbose' : False, # print PVGIS parameters and requests
#    'gen_neg' : False, # return negative P
#}

# For each PV
#PVMODEL_PARAMS = {
#    'scale_factor' : 10000,
#    'lat' : 52.373,
#    'lon' : 9.738,
#    'optimal_both' : True,
#}

# Wind power
#WECS_CONFIG = {'P_rated': 7000, 
#               'v_rated': 13, 
#               'v_min': 3.5, 
#               'v_max': 25}

def input_to_state(aeid, aid, input_data, current_state, current_time, first_time_step, **kwargs):
    # aeid: Agent_3
    # aid: agent3
    # input_data: {'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0, 'FLSim-0.FLSim-0': 0.9}}
    # current_state: {'production': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}, 
    #                'consumption': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}}
#    global cells
    global profiles
    state = copy.deepcopy(MAS_STATE)
    #print(aeid, input_data)
    for eid, value in input_data.items():
        value = sum(value.values())
        print(current_time, eid, value)
        if pd.isna(value):
            value = 0
        if 'Load' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption'] = update_flexibility(state['consumption'], profile)
                state['consumption']['current'] = value
                #print('Load', state)
#        elif 'PV' in eid or 'WECS' in eid:
#                if first_time_step:
#                    profile['max'] = min(abs(value), profile['max'])
#                value = np.clip(abs(value), profile['min'], profile['max'])
#                state['production'] = update_flexibility(state['production'], profile)
#                state['production']['current'] = value
#                #print('Gen', state)
        elif 'StaticGen' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                value = np.clip(abs(value), profile['min'], profile['max'])
                state['production'] = update_flexibility(state['production'], profile)
                state['production']['current'] = value
#                #print('Gen', state)
        elif 'ExternalGrid' in eid:
                profile = get_unit_profile(eid, current_time, profiles)
                if not len(profile):
                    profile['min'] = 5
                    profile['max'] = 5
                state['production'] = update_flexibility(state['production'], profile)
                state['consumption'] = update_flexibility(state['consumption'], profile)
                if value > 0: # check the convention here!
                    state['production']['current'] = value
                else:
                    state['consumption']['current'] = abs(value) 
        break 
#
    state['consumption']['scale_factor'] = current_state['consumption']['scale_factor']
    state['production']['scale_factor'] = current_state['production']['scale_factor']
    return state

# Multi-agent system (MAS) configuration
# User-defined methods are specified in methods.py to make scenario concise, 
# except `input_method`, since it requieres an access to global variables
MAS_CONFIG = { # see MAS_DEFAULT_CONFIG in utils.py 
    # Required parameters
    'META': MAS_META,
    'verbose': args.verbose, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': args.performance, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'convergence_steps' : 1, # higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence
    'state_dict': MAS_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'aggregation_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'execute_method': execute_instructions,    # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers,
                                               # executes the received instructions internally
    'initialize' : initialize,
    'finalize' : finalize,#

    # Additional user-defined parameters
    'between-cells' : 'default', #'default', 'cohda'
    'within-cell' : 'cohda', 
}

world = mosaik.World(SIM_CONFIG)
with world.group():
        massim = world.start('MASSCA', 
                            step_size=STEP_SIZE, 
                            **MAS_CONFIG,
                            )
        #pvsim = world.start(
        #                "PVSim",
        #                step_size=STEP_SIZE,
        #                sim_params=PVSIM_PARAMS,
        #            )
        flsim = world.start(
                        "FlexSim",
                        step_size=STEP_SIZE,
                        #csv_file=LOAD_FILE,
                        sim_params=dict(gen_neg=False),
                    )
        #wsim = world.start('WPSim', 
        #                step_size=STEP_SIZE, 
        #                wind_file=WIND_FILE,
        #            )
    
input_sim = world.start("InputSim", 
                              sim_start=START_DATE, 
                              date_format=DATE_FORMAT,
                              datafile=prof_file)
inputs = input_sim.Profiles.create(1)[0]
    
output_sim = world.start('OutputSim', start_date = START_DATE,
                                          output_file=os.path.join(results_dir, args.output_file))
outputs = output_sim.CSVWriter(buff_size=STEP_SIZE)
    
grid_sim = world.start('GridSim', step_size=STEP_SIZE)
units = pd.concat([grid.load, grid.sgen, grid.ext_grid, grid.bus], 
                        ignore_index=True).set_index('name')
grid = grid_sim.Grid(json=grid_file)
units = {k : (e, v.to_dict()) for k, v in units.iterrows() #grid_sim.get_extra_info().items()
                            for e in grid.children
                                if e.eid == k}

    
#print(profiles)
#sys.exit()

agents = [] # one simple agent per unit (sgen, load)
controllers = [] # top-level agents that are named as cell_controllers, one per cell
hierarchical_controllers = []

mosaik_agent = massim.MosaikAgents() # core agent for the mosaik communication
world.connect(units['ExternalGrid-0'][0], outputs, ("P[MW]", "value"))
world.connect(units['ExternalGrid-0'][0], mosaik_agent, ('P[MW]', 'ExternalGrid-0.value'), 
                                            time_shifted=True, initial_data={'P[MW]': 0})
world.connect(mosaik_agent, outputs, ("production[MW]", "ExternalGrid"))

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
            
            #world.connect(fload, units[f"Bus-{int(v[1]['bus'])}"][0], ("P[MW]", "P_load[MW]"))
            #world.connect(fload, outputs, ("P[MW]", f"{k}-current"))
            #world.connect(inputs, units[f"Bus-{int(v[1]['bus'])}"][0], (f"{k}.current", "P_load[MW]"))

            #world.connect(fload, units[f"Bus-{int(v[1]['bus'])}"][0], ("P[MW]", "P_load[MW]"))
            #world.connect(fload, outputs, ("P[MW]", f"{k}-current"))
            
            world.connect(agents[-1], outputs)
            switch_off.append(k)
            
        elif "StaticGen" in k:
            fgen = flsim.FLSim.create(num=1)[0]
            agents += massim.MosaikAgents.create(num=1, controller=hierarchical_controllers[cell_idx].eid)
            world.connect(inputs, fgen, (f"{k}.value", "P[MW]"))
            world.connect(fgen, agents[-1], ('P[MW]', f"{k}.production[MW]"))
            world.connect(agents[-1], fgen, ('production_delta[MW]', 'scale_factor'), weak=True)
            world.connect(agents[-1], units[f"Bus-{int(v[1]['bus'])}"][0], ("production[MW]", "P_gen[MW]"))
            
            #world.connect(fgen, units[f"Bus-{int(v[1]['bus'])}"][0], ("P[MW]", "P_gen[MW]"))
            #world.connect(fgen, outputs, ("P[MW]", f"{k}-current"))
            #world.connect(inputs, units[f"Bus-{int(v[1]['bus'])}"][0], (f"{k}.current", "P_gen[MW]"))
            
            
            
            world.connect(agents[-1], outputs)      
            switch_off.append(k)
            
        elif "Bus" in k:
            world.connect(v[0], outputs, ("P[MW]", "value"))
            pass
            
grid_sim.disable_elements(switch_off)
      
    
    #print(grid_sim.get_extra_info())
    #world.connect(v[0], outputs, ("P[MW]", "current"))
    #print(loads)
    #print(buses)
    #print(gens)
    #print(ext_grids)
    
    
    
#if args.performance:
#    mosaik.util.plot_dataflow_graph(world, hdf5path=os.path.join(results_dir, '.hdf5'), show_plot=False)
#world.run(until=END, print_progress='individual' if args.performance else True)
world.run(until=END)
sys.exit()
    
'''
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

    for i in [i for i in cells.keys() if 'match' not in i]:
        cell_controllers += masim.MosaikAgents.create(num=1, controller=None)
        #world.connect(cell_controllers[-1], report, 'current')

        entities = list(cells[i]['StaticGen'].values()) + list(cells[i]['Load'].values())
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
                        #connected_gens += 1
                            #print(e)
                    elif e['type'] == 'Load':
                            agents += masim.MosaikAgents.create(num=1, controller=hierarchical[-1].eid)
                            e.update({'agent' : agents[-1], 'sim' : inputs})   
                            world.connect(e['sim'], bus, (e['name'], 'P_load[MW]'))
                            world.connect(e['sim'], e['agent'], e['name'])#(e['name'], 'current'))
                            world.connect(e['sim'], report, e['name'])#(e['name'], 'P[MW]'))
                            #connected_loads += 1

                            



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
    '''
if __name__ == '__main__':
    main()
