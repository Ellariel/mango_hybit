#!/usr/bin/env python
# coding: utf-8
# Import packages needed for the scenario
import pandas as pd
import h5py
import mosaik
#import mosaik_api_v3 as mosaik
import mosaik_components
#from mosaik.util import connect_randomly, connect_many_to_one
from mosaik.util import connect_many_to_one
import warnings
#from mosaik.scheduler import warn_if_successors_terminated
warnings.filterwarnings("ignore", category=FutureWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x


sim_config = {
    'CSV': {
        'python': 'mosaik_csv:CSV',
    },
    'ControlledGen': {
        'python': 'mosaik_components.pandapower:Simulator'
    },
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator'
    },
    'DB': {
         'cmd': 'mosaik-hdf5 %(addr)s'
    },
    'Emission': {
         'python': 'emission_simulator:EmissionSimulator'
    },
}

mosaik_config = {
    'warn_if_successors_terminated' : False,
}

emission_config = {
    "Gen": {
        "input_attr": "P[MW]",
        "output_attr": "E[tCO2eq]", 
        "fuel": "Natural Gas Liquids",
        #"co2_default_emission_factor": 64200 / 1000 / 0.2778 / 1000, 
        #[kg CO₂eq. / TJ] -> [tones CO₂eq. / TJ] -> (1 TJ = 0.2778 GWh) -> [tones CO₂eq. / MWh],
        "method": None,
    },
    "ExternalGrid": {
        "input_attr": "P[MW]",
        "output_attr": "E[tCO2eq]",
        "country": "Germany",
        #"co2_intensity": 385 / 1000, 
        #[grams CO₂eq. / kWh] -> [kg CO₂eq. / MWh] -> [tones CO₂eq. / MWh]
        "method": None,
    },

}

# Add PV data file
END = 60*60 #* 60 #1 * 24 * 60 * 60  #two days
START = '2014-01-01 01:00:00'
GRID_FILE = 'pandapower_example.json' # import directly from pandpaower simbench module

# Set up the "world" of the scenario.
world = mosaik.World(sim_config, mosaik_config)

# Initialize the simulators.
csv_sim = world.start('CSV', 
                      sim_start=START,
                      datafile='./data/example_data.csv',
                      #date_format='YYYY-MM-DD HH:mm:ss',
                      delimiter=',')

#gensim = world.start('ControlledGen')#,

gridsim = world.start('Grid', step_size=60*60)#,
                      #sim_params={"numba": False})#, 
                      #step_size=60*60, 
                      #mode = 'pf_timeseries') # mode = 'pf' or 'pf_timeseries'

db_sim = world.start('DB', 
                 step_size=60, 
                 duration=END)

em_sim = world.start('Emission', sim_params=emission_config)

# Instantiate model entities.
#print(csv_sim['DNI'])
csv = csv_sim.TestData.create(1)

grid = gridsim.Grid(json=GRID_FILE)#,
                    #numba=False)#, 
                    #sim_start=START)

gen = gridsim.ControlledGen(bus=0)#.create(1)

em = em_sim.Emission.create(1)

print(grid.children)
hdf5 = db_sim.Database(filename='ppw.hdf5')

buses = [e for e in grid.children if e.type in ['Bus']]
gens = [e for e in grid.children if e.type in ['Gen', 'ControlledGen']]
ext_grid = [e for e in grid.children if e.type in ['ExternalGrid']]

print(gen.eid)
print(world.entity_graph)

world.connect(csv[0], gen, ('P', 'P[MW]')) #buses[0]
connect_many_to_one(world, buses, hdf5, ('P[MW]', 'p_mw'), ('Vm[pu]', 'vm_pu')) #('Q[MVar]', 'q_mw'), ('Va[deg]', 'va_deg')
connect_many_to_one(world, em, hdf5, 'E[tCO2eq]', 'P[MW]') #('Q[MVar]', 'q_mw'), ('Va[deg]', 'va_deg')

world.connect(gens[0], em[0], ('P[MW]', 'P[MW]'))
world.connect(ext_grid[0], em[0], ('P[MW]', 'P[MW]'))

#connect_many_to_one(world, buses, hdf5, ('P[MW]', 'p_mw'), ('Vm[pu]', 'vm_pu'))

#Run
world.run(until=END)
world.shutdown()

#filename = "ppw.hdf5"

#with h5py.File(filename, "r") as f:
#    print("Keys: %s" % f.keys())
#    for k in f.keys():
#        print(f[k])
#        print(type(f[k]))
       



