import os
import json
import arrow
import random
import pandas as pd
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.toolbox import merge_nets

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

'''
Cell based grid generating script

'''

def set_random_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_random_seed(base_seed=None, fixed_range=1000):
    if base_seed:
        set_random_seed(base_seed)
    return random.randint(0, fixed_range)

def get_cell():
    '''
    https://pandapower.readthedocs.io/en/v2.10.0/networks/cigre.html#medium-voltage-distribution-network-with-all-der
    This pandapower network includes the following parameter tables:
    8 photovoltaic generators
    1 wind turbine
    2 Batteries
    2 residential fuel cells
    1 CHP diesel
    1 CHP fuel cell
    '''
    def _switch(net, type='CB', et='t', element=1, bus=0, closed=False): 
        net.switch.loc[(net.switch['type'] == type) &
                       (net.switch['et'] == et) &
                       (net.switch['element'] == element) &
                       (net.switch['bus'] == bus), 'closed']  = closed
    
    net = pn.create_cigre_network_mv(with_der="all") # gets a cell subnetwork
    _switch(net, closed=False) # switching off the right part (trafo 1, see the scheme) of the subnetwork
    return net

def create_cells(cells_count=2, dir='./', validation=False):
    '''
    Creates a network that consists of ´cells_count´ connected CIGRE subnetworks.
    Saves the network json file `cells.json` to `dir`.
    '''
    def _rename(elements, cutoff_index, ren_type=None): # create new name for each unit 'type-index-bus-cell'
        for idx, g in elements.iterrows():
            new_type = g.type.lower() if not ren_type else ren_type
            elements.loc[elements.index == idx, 'type'] = new_type
            elements.loc[elements.index == idx, 'name'] = f"{new_type}-{idx}-{g.bus}-{int(g.bus / cutoff_index) if cutoff_index > 0 else 0}"

    def _save_old_names(elements, mapping, ren_type=None):
        for idx, g in elements.iterrows():
            new_type = g.type.lower() if not ren_type else ren_type
            if idx in mapping:
                elements.loc[elements.index == idx, 'type'] = new_type
                elements.loc[elements.index == idx, 'name'] = f"{new_type}-{idx}-{idx}" 
            else:  
                for old_idx, new_idx in mapping.items():
                    if idx == new_idx:
                        elements.loc[elements.index == idx, 'type'] = new_type
                        elements.loc[elements.index == idx, 'name'] = f"{new_type}-{idx}-{old_idx}"
                        break

    cell_cutoff = 0
    net = get_cell()
    for _ in range(cells_count-1):
        new_subnet = get_cell()
        cell_cutoff = len(new_subnet.bus) # identify the cell index based on the index of buses
        net, idx = merge_nets(net, new_subnet, validation=False, numba=False, return_net2_reindex_lookup=True)
        _save_old_names(net.bus, idx['bus'], ren_type='Bus')
        pp.create_line(net, from_bus=0, to_bus=idx['bus'][0], length_km=5, std_type="NAYY 4x50 SE") # connect merged cells
        net.ext_grid.drop(1, inplace=True) # drop excessive external grid
        net.res_ext_grid.drop(1, inplace=True)
    _rename(net.sgen, cell_cutoff, ren_type='StaticGen')
    _rename(net.load, cell_cutoff, ren_type='Load')
    _rename(net.ext_grid, cell_cutoff, ren_type='ExternalGrid')

    if validation:
        pp.runpp(net, numba=False) # to test
    if dir:
        dir = os.path.join(dir, "cells.json")
        pp.to_json(net, dir)
    return net, dir

def _randomize_timeseries(start='2016-01-01 00:00:00', end=60*15*1, step_size=60*15, target_value_min=0.01, target_value_max=2, name='FLSim', dir='./', seed=13):
    random.seed(seed)
    np.random.seed(seed)
    date_start = arrow.get(start, 'YYYY-MM-DD hh:mm:ss')
    date_end = date_start.shift(seconds=end)
    date_list = [i.format('YYYY-MM-DD HH:mm:ss') for i in list(arrow.Arrow.range('minute', date_start, date_end))[::int(step_size/60)]]
    d = pd.Series(date_list).rename('Time') #pd.to_datetime(date_list, format='mixed')
    loads = [pd.Series([random.uniform(target_value_min, target_value_max) for i in d]).rename(name)]
    data = pd.concat([d] + loads, axis=1).to_dict() #.set_index('Time')
    #print(data)
    return data

def generate_profiles(net, dir='./', start='2016-01-01 00:00:00', end=60*15*1, step_size=60*15, seed=13):
    profiles = {}
    random.seed(seed)
    np.random.seed(seed)
    for _, unit in net.sgen.iterrows():
        p = unit.p_mw #* random.randrange(1, 10)
        #print(unit['name'], p)
        profiles[unit['name']] = {
                'max' : p,
                'min' : 0,
                'current' : _randomize_timeseries(start=start, end=end, name=unit['name'], dir=dir, seed=seed, target_value_min=0, target_value_max=p)
            }
        #print(profiles[unit['name']])
    for _, unit in net.load.iterrows():
        p = unit.p_mw #* random.randrange(1, 4)
        p_max = (p + p * random.randrange(1, 95) / 100)
        profiles[unit['name']] = {
                'max' : p_max,
                'min' : p, 
                'current' : _randomize_timeseries(start=start, end=end, name=unit['name'], dir=dir, seed=seed, target_value_min=p, target_value_max=p_max)
            }
        #print(unit['name'], profiles[unit['name']])
    for _, unit in net.res_ext_grid.iterrows():
        p = unit.p_mw
        profiles[net.ext_grid.iloc[unit.name]['name']] = {
                'max' : p * 10,
                'min' : 0,
            }
    if dir:
        dir = os.path.join(dir, "profiles.json")
        with open(dir, 'w') as f:
            json.dump(profiles, f)
    return profiles, dir


def get_cells_data(grid, grid_extra_info, profiles):
    def lookup_bus_name(idx):
        return grid_extra_info[f'Bus-{idx}']['name']

    cells = {}
    for e in grid.children:
        if e.eid in grid_extra_info and\
           'name' in grid_extra_info[e.eid] and\
           pd.notna(grid_extra_info[e.eid]['name']):
                name = grid_extra_info[e.eid]['name']
                id = name.split('-')
                if len(id) == 4: # type-index-bus-cell
                    cells.setdefault(id[3], {})
                    cells.setdefault('match_cell', {})
                    cells['match_cell'].update({e.eid : id[3]})
                    cells[id[3]].setdefault(id[0], {})
                    cells[id[3]][id[0]].update({e.eid : {
                        'name' : name,
                        'unit' : e,
                        'type' : id[0],
                        'index' : id[1],
                        'bus' : lookup_bus_name(id[2]),
                        'cell' : id[3],
                        'profile' : profiles[name] if name in profiles else {},
                    }})
    return cells

def get_unit_profile(aeid, cells_data):
    if aeid == 'MosaikAgent':
        unit_eid = cells_data['match_agent'][aeid]
        unit_type = unit_eid.split('-')[0]
        return cells_data[cells_data['match_cell'][unit_eid]][unit_type][unit_eid]['profile']
    if aeid in cells_data['match_agent']:
        sim_eid = cells_data['match_agent'][aeid]
        if sim_eid in cells_data['match_unit']:
            unit_eid = cells_data['match_unit'][sim_eid]
            unit_type = unit_eid.split('-')[0]
            if unit_eid in cells_data['match_cell']:
                return cells_data[cells_data['match_cell'][unit_eid]][unit_type][unit_eid]['profile']
    return {}

if __name__ == '__main__':
    cells_count = 2
    net, net_file = create_cells(cells_count=cells_count)
    print(f'The network of {cells_count} cells is created and saved at: {net_file}')
    print(net)
    profiles, prof_file = generate_profiles(net)
    print(f'The profiles were generated and saved at: {prof_file}')
    print(profiles)