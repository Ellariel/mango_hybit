"""
This file contains the mosaik scenario.  To start the simulation, just run this
script from the command line::

    $ python scenario.py

Since neither the simulator in ``src/wecssim`` nor the MAS in ``src/mas`` are
installed correctly, we add the ``src/`` directory to the PYTHONPATH so that
Python will find these modules.


"""
import sys
import mosaik
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandapower.auxiliary
pandapower.auxiliary._check_if_numba_is_installed = lambda x: x

#sys.path.insert(0, 'src/')

SIM_CONFIG = {
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator'
    },
    'WecsSim': {
        'python': 'wecssim.mosaik:WecsSim',
    },
    'MAS': {
        'python': 'mosaik_agents:MosaikAgents'
    },
    #'DB': {
    #    'cmd': 'mosaik-hdf5 %(addr)s',
    #},
}

DURATION = 3600 #* 24 * 1  # 1 day
START_DATE = '2014-01-01T00:00:00+01:00'  # CET
GRID_FILE = 'data/pandapower_example.json' 
WIND_FILE = 'data/wind_speed_m-s_15min.csv'
DB_PATH = 'data/mosaik_results.hdf5'

WECS_CONFIG = [
    #(1, {'P_rated': 2000, 'v_rated': 12, 'v_min': 2.0, 'v_max': 25,  'controller': None}),
    (1, {'P_rated': 5000, 'v_rated': 13, 'v_min': 3.5, 'v_max': 25, 'controller': None}),
]

AGENTS_CONFIG = [
    (4, {'controller': None, 'initial_state': {}}),
]
CONTROLLERS_CONFIG = [
    (1, {'controller': None, 'initial_state': {}}),
]

def main():
    """Compose the mosaik scenario and run the simulation."""
    world = mosaik.World(SIM_CONFIG)
    wecssim = world.start('WecsSim', step_size=60*15, wind_file=WIND_FILE)
    gridsim = world.start('Grid', step_size=60*15)
    mas = world.start('MAS')
    #db = world.start('DB', step_size=60*60, duration=DURATION)

    grid = gridsim.Grid(json=GRID_FILE)
    print(grid.children)
    buses = [e for e in grid.children if e.type in ['Bus']]
    gens = [e for e in grid.children if e.type in ['Gen', 'ControlledGen']]
    ext_grids = [e for e in grid.children if e.type in ['ExternalGrid']]
    loads = [e for e in grid.children if e.type in ['Load']]
    #hdf5 = db.Database(filename=DB_PATH)  

    controllers = []
    for n, params in CONTROLLERS_CONFIG:  # Iterate over the config sets
        controllers += mas.MosaikAgents.create(num=n, **params)
    print('controllers:', controllers)

    agents = []
    for n, params in AGENTS_CONFIG:  # Iterate over the config sets
        params.update({'controller' : controllers[0].eid})
        agents += mas.MosaikAgents.create(num=n, **params)
    print('agents:', agents)

    wecs = []
    for n_wecs, params in WECS_CONFIG:  # Iterate over the config sets
        for _ in range(n_wecs):
            w = wecssim.WECS(**params)
            wecs.append(w)
    print('wecs:', wecs)

    world.connect(wecs[0], agents[0], 'P', async_requests=True)
    world.connect(gens[0], agents[1], 'P[MW]', async_requests=True)
    world.connect(loads[0], agents[2], 'P[MW]', async_requests=True)
    world.connect(ext_grids[0], agents[3], 'P[MW]', async_requests=True)

    #world.connect(wecs[0], controllers[0], 'P', async_requests=True)
    #world.connect(controllers[0], hdf5, 'P', async_requests=True)
    #mosaik.util.connect_many_to_one(world, wecs, hdf5, 'v', 'P', 'P_max')

    world.run(DURATION)


if __name__ == '__main__':
    main()
