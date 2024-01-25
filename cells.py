import os
import json
import random
import numpy as np
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as pt
import pandapower.topology as tp
import matplotlib.pyplot as plt
from pandapower.toolbox import merge_nets

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

'''
Cells based grid generation

'''

def get_cell():
    # https://pandapower.readthedocs.io/en/v2.10.0/networks/cigre.html#medium-voltage-distribution-network-with-pv-and-wind-der
    '''
    This pandapower network includes the following parameter tables:
    - switch (8 elements)
    - load (18 elements)
    - ext_grid (1 elements)
    - sgen (9 elements)
    - line (15 elements)
    - trafo (2 elements)
    - bus (15 elements)
    - bus_geodata (15 elements)
    '''
    return pn.create_cigre_network_mv(with_der="pv_wind")

def create_cells(cells_count=2, dir='./', validation=False):
    def rename(elements, cutoff_index, ren_type=None): # create new name for each unit 'type-index-bus-cell'
        for idx, g in elements.iterrows():
            new_type = g.type.lower() if not ren_type else ren_type
            elements.loc[elements.index == idx, 'type'] = new_type
            elements.loc[elements.index == idx, 'name'] = f"{new_type}-{idx}-{g.bus}-{int(g.bus / cutoff_index) if cutoff_index > 0 else 0}"

    cell_cutoff = 0
    net = get_cell()
    for _ in range(cells_count-1):
        new_subnet = get_cell()
        cell_cutoff = len(new_subnet.bus) # identify the cell index based on the index of buses
        net, idx = merge_nets(net, new_subnet, validation=False, numba=False, return_net2_reindex_lookup=True)
        pp.create_line(net, from_bus=0, to_bus=idx['bus'][0], length_km=5, std_type="NAYY 4x50 SE") # connect merged cells
        net.ext_grid.drop(1, inplace=True) # drop excessive external grid
        net.res_ext_grid.drop(1, inplace=True)
    rename(net.sgen, cell_cutoff)
    rename(net.load, cell_cutoff, ren_type='load')
    if validation:
        pp.runpp(net, numba=False) # to test
    if dir:
        dir = os.path.join(dir, "cells.json")
        pp.to_json(net, dir)
    return net, dir

def generate_profiles(net, dir='./', seed=13):
    profiles = {}
    random.seed(seed)
    np.random.seed(seed)
    for _, unit in net.sgen.iterrows():
        profiles[unit['name']] = {
                'max' : unit.p_mw,
                'min' : 0
            }
    for _, unit in net.load.iterrows():
        profiles[unit['name']] = {
                'max' : unit.p_mw * random.randrange(1, 4),
                'min' : (unit.p_mw - unit.p_mw * random.randrange(1, 4) / 10) * random.randrange(0, 2),
            }
    if dir:
        dir = os.path.join(dir, "profiles.json")
        with open(dir, 'w') as f:
            json.dump(profiles, f)
    return profiles, dir

if __name__ == '__main__':
    cells_count = 2
    net, net_file = create_cells(cells_count=cells_count)
    print(f'The network of {cells_count} cells is created and saved at: {net_file}')
    print(net)
    profiles, prof_file = generate_profiles(net)
    print(f'The profiles were generated and saved at: {prof_file}')
    print(profiles)
