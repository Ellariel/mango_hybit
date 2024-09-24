import os
import sys
import time
import json
from pathlib import Path
import argparse
import pandapower as pp
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
#parser.add_argument('--cells', default=2, type=int)
#parser.add_argument('--verbose', default=0, type=int)
#parser.add_argument('--clean', default=True, type=bool)
parser.add_argument('--dir', default='./', type=str)
#parser.add_argument('--seed', default=13, type=int)
#parser.add_argument('--output_file', default='results.csv', type=str)
#parser.add_argument('--performance', default=True, type=bool)
#parser.add_argument('--hierarchy', default=1, type=int)
args = parser.parse_args()

base_dir = args.dir
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
grid_file = os.path.join(data_dir, 'grid_model.json')

net = pp.create_empty_network()
# https://pandapower.readthedocs.io/en/latest/topology/examples.html

pp.create_bus(net, name = "Bus-0", vn_kv = 110, type = 'b') # bus 0, 110 kV bar
pp.create_bus(net, name = "Bus-1", vn_kv = 20, type = 'b') # bus 1, 20 kV bar
pp.create_bus(net, name = "Bus-2", vn_kv = 20, type = 'b')
pp.create_bus(net, name = "Bus-3", vn_kv = 20, type = 'b')
pp.create_bus(net, name = "Bus-4", vn_kv = 20, type = 'b')

pp.create_ext_grid(net, 0, vm_pu = 1, name = "ExternalGrid-0")

pp.create_line(net, name = "Line-0", from_bus = 1, to_bus = 2, length_km = 1, std_type = "NAYY 4x150 SE")
pp.create_line(net, name = "Line-1", from_bus = 1, to_bus = 3, length_km = 1, std_type = "NAYY 4x150 SE")
pp.create_line(net, name = "Line-2", from_bus = 1, to_bus = 4, length_km = 1, std_type = "NAYY 4x150 SE")

pp.create_transformer_from_parameters(net, hv_bus=0, lv_bus=1, i0_percent=0.038, pfe_kw=11.6,
        vkr_percent=0.322, sn_mva=40, vn_lv_kv=22.0, vn_hv_kv=110.0, vk_percent=17.8)

pp.create_load(net, 2, p_mw = 1, q_mvar = 0.2, name = "Load-0")
pp.create_load(net, 4, p_mw = 1, q_mvar = 0.2, name = "Load-1")

pp.create_sgen(net, 3, p_mw = 1, name = "StaticGen-0")

pp.runpp(net, numba=False)

pp.to_json(net, grid_file)

print("buses", net.bus)
print("loads", net.load)
print("sgens", net.sgen)
print("ext_grid", net.ext_grid)

print(f"Grid model was saved: {grid_file}")