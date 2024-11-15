import sys, os, copy
import numpy as np
import nest_asyncio
import logging
import asyncio
import random

from ..utils import make_hashable
old_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
from mosaik_cohda.cohda_simulator import Simulator, Flexibility, StartValues
sys.stdout = old_stdout

        
class COHDA():
    def __init__(self, step_size=15*60, time_resolution=1., 
                 host='localhost', base_port=7060, verbose=True, **sim_params):
        self.step_size = step_size
        self.sim_params = sim_params
        self.time_resolution = time_resolution
        self.host = host
        self.verbose = verbose
        self.old_stdout = sys.stdout
        if not self.verbose:
            logging.disable()
            logging.shutdown()
        self.base_port = base_port
        self.cache = {}
        nest_asyncio.apply()

    def reinitialize(self, n_agents):
        simulator = Simulator()
        simulator.id = random.randint(1, 10)
        simulator.host = self.host
        simulator.port = self.base_port + simulator.id
        self.base_port += n_agents
        self.sim_params.update({'loop': asyncio.new_event_loop()})
        simulator.init(sid=simulator.id, step_size=self.step_size, 
                       time_resolution=self.time_resolution, **self.sim_params)
        agent_params = {'control_id': 0, 
                'time_resolution': self.time_resolution,
                'step_size' : self.step_size}
        simulator.agents = []
        for i in range(n_agents):
            agent_model = simulator.create(1, 'FlexAgent', **agent_params)
            simulator.agents.append(agent_model)
        simulator.setup_done()
        return simulator

    def execute(self, target_schedule, flexibility, **kwargs):
            seed = kwargs.get('seed', None)
            cache_key = make_hashable((target_schedule, flexibility, seed))
            if cache_key in self.cache:
                if self.verbose:
                    print('COHDA returns a cached solution.')
                return copy.deepcopy(self.cache[cache_key])
            
            random.seed(seed)
            np.random.seed(seed)

            if not self.verbose:
                sys.stdout = open(os.devnull, "w")
           
            n_agents = len(flexibility)
            participants = list(range(n_agents))
            simulator = self.reinitialize(n_agents=n_agents)
            input_data = {}
            output_data = {}
            simulator.uids = {}
            for i in participants:
                agent, flex = simulator.agents[i], flexibility[i]
                flex_max_energy_delta = flex.get('flex_max_energy_delta', [100] * len(flex['flex_max_power']))
                flex_min_energy_delta = flex.get('flex_min_energy_delta', [0] * len(flex['flex_min_power']))
                eid = agent[0]['eid']
                data = {'StartValues': {'ID_0': StartValues(schedule=target_schedule, 
                                                            participants=participants)},
                        'Flexibility': {'ID_0': Flexibility(flex_max_power=flex['flex_max_power'],
                                                            flex_min_power=flex['flex_min_power'],
                                                            flex_max_energy_delta=flex_max_energy_delta,
                                                            flex_min_energy_delta=flex_min_energy_delta)}
                }
                input_data[eid] = data
                simulator.uids[eid] = None
                output_data[eid] = ['FlexSchedules']

            simulator.step(time=0, inputs=input_data, max_advance=0)
            simulator.get_data(outputs=output_data)
            schedules = [simulator.schedules[agent[0]['eid']]['FlexSchedules'] for agent in simulator.agents]
            self.cache[cache_key] = schedules
            simulator.finalize()
            del simulator
            
            if not self.verbose:
                sys.stdout = self.old_stdout

            return copy.deepcopy(self.cache[cache_key])

