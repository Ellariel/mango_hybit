import pandapower as pp
'''
Cells grid

'''

import pandapower.networks as pn

net = pn.create_cigre_network_mv(with_der="pv_wind")
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

pp.to_json(net, "./cells_example.json")

# test
pp.runpp(net, numba=False)
print('bus', net.res_bus)#.vm_pu)
print('line', net.res_line)#.loading_percent)
print('gen', net.res_gen)
print('load', net.res_load)