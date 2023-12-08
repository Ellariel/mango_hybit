import pandapower as pp
'''
Simple grid

'''
# create empty net
net = pp.create_empty_network() 

# create buses
b1 = pp.create_bus(net, vn_kv=20.)
b2 = pp.create_bus(net, vn_kv=20.)

# create elements
line = pp.create_line(net, from_bus=b1, to_bus=b2, length_km=2.5, std_type="NAYY 4x50 SE")   
ext_grid = pp.create_ext_grid(net, bus=b1)
gen = pp.create_gen(net, bus=b1, p_mw=1.)#, slack=True)
load = pp.create_load(net, bus=b2, p_mw=1.)

pp.to_json(net, "./data/pandapower_example.json")

# test
pp.runpp(net, numba=False)
print('bus', net.res_bus)#.vm_pu)
print('line', net.res_line)#.loading_percent)
print('gen', net.res_gen)
print('load', net.res_load)