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

cells_count = 2

cells_count = 2

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

net = get_cell()
for i in range(cells_count):
    new_subnet = get_cell()
    net, idx = merge_nets(net, new_subnet, validation=False, numba=False, return_net2_reindex_lookup=True)
    pp.create_line(net, from_bus=0, to_bus=idx['bus'][0], length_km=5, std_type="NAYY 4x50 SE") # connect merged cells
    net.ext_grid.drop(1, inplace=True) # drop excessive external grid
    net.res_ext_grid.drop(1, inplace=True)
    
pp.runpp(net, numba=False)
pp.to_json(net, "./cells_net.json")

# test
#pp.runpp(net, numba=False)
#print('bus', net.res_bus)#.vm_pu)
#print('line', net.res_line)#.loading_percent)
#print('gen', net.res_gen)
#print('load', net.res_load)