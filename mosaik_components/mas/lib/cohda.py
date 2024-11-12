import sys, os, copy
old_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
import logging
import asyncio
import random
import nest_asyncio
import mosaik_api_v3 as mosaik_api
from mango import Agent, RoleAgent
from mosaik_cohda.des_flex.flex_class import Flexibility
from mosaik_cohda.start_values import StartValues
from mosaik_cohda.cohda_simulator import CohdaSimulator
from mosaik_cohda.agent_roles import FlexReceiverRole, FlexCohdaRole, \
    FlexTerminationRole, FlexNegotiationStarterRole
from mosaik_cohda.start_values import StartValues, SolutionSchedule
from mosaik_cohda.mango_library.coalition.core import CoalitionParticipantRole
from mosaik_cohda.cohda_simulator import logger
from mosaik_components.mas.utils import *

sys.stdout = old_stdout

class FlexReceiverRoleModified(FlexReceiverRole):
    def setup(self) -> None:
        super().setup()
        self.id = int(self.context.aid.split('agent')[1]) + 1

    def on_change_model(self, model) -> None:
        if model.terminated:
            schedule = self.context.data.current_cohda_solution.solution_candidate.candidate[self.id]
            solution = SolutionSchedule(schedule)
            self.context.schedule_instant_task(
                self.context.send_acl_message(
                    receiver_addr=self._mosaik_agent_address[0],
                    receiver_id=self._mosaik_agent_address[1],
                    content=solution,
                    acl_metadata={'conversation_id':
                                      self._mosaik_agent_address[1],
                                  'sender_id': self.context.aid},
                )
            )

class Simulator(CohdaSimulator):

    async def create_flex_agent(self, container,
                                time_resolution, initiator: bool = False) -> Agent:
        a = RoleAgent(container)
        a.add_role(CoalitionParticipantRole())
        a.add_role(FlexCohdaRole(time_resolution=time_resolution))
        a.add_role(FlexReceiverRoleModified((self.main_container.addr,
                                     self.mosaik_agent.aid)))
        a.add_role(FlexTerminationRole(initiator))
        if initiator:
            a.add_role(FlexNegotiationStarterRole())
        return a

    @staticmethod
    async def _shutdown(*args):
        futs = []
        for arg in args:
            futs.append(arg.shutdown())
        await asyncio.gather(*futs)

    def finalize(self):
        try:
            if not self.loop.is_closed():
                self.loop.run_until_complete(
                    self._shutdown(self.mosaik_agent,
                                self.agent_container,
                                self.main_container
                                ))
        except:
            pass
        
class COHDA():
    def __init__(self, step_size=15*60, time_resolution=1., host='localhost', base_port=7060, muted=False, **sim_params):
        self.step_size = step_size
        self.sim_params = sim_params
        self.time_resolution = time_resolution
        self.host = host
        self.muted = muted
        self.old_stdout = sys.stdout
        if self.muted:
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
        simulator.init(sid=simulator.id, step_size=self.step_size, time_resolution=self.time_resolution, **self.sim_params)
        agent_params = {'control_id': 0, 
                'time_resolution': self.time_resolution,
                'step_size' : self.step_size}
        simulator.agents = []
        for i in range(n_agents):
            agent_model = simulator.create(1, 'FlexAgent', **agent_params)
            simulator.agents.append(agent_model)
        simulator.setup_done()
        return simulator

    def execute(self, target_schedule, flexibility):

            cache_key = make_hash_sha256(make_hashable((target_schedule, flexibility)))
            if cache_key in self.cache:
                if not self.muted:
                    print('COHDA returns a cached solution.')
                return copy.deepcopy(self.cache[cache_key])

            if self.muted:
                sys.stdout = open(os.devnull, "w")
           
            n_agents = len(flexibility)
            participants = list(range(n_agents))
            simulator = self.reinitialize(n_agents=n_agents)
            input_data = {}
            output_data = {}
            simulator.uids = {}
            for i in participants:
                agent, flex = simulator.agents[i], flexibility[i]
                eid = agent[0]['eid']
                data = {'StartValues': {'ID_0': StartValues(schedule=target_schedule, 
                                                            participants=participants)},
                        'Flexibility': {'ID_0': Flexibility(flex_max_power=flex['flex_max_power'],
                                                            flex_min_power=flex['flex_min_power'],
                                                            flex_max_energy_delta=[100] * len(flex['flex_min_power']),
                                                            flex_min_energy_delta=[0] * len(flex['flex_min_power']),
                                                        )}
                }
                input_data[eid] = data
                simulator.uids[eid] = None
                output_data[eid] = ['FlexSchedules']

            simulator.step(time=0, inputs=input_data, max_advance=0)
            simulator.get_data(outputs=output_data)
            self.cache[cache_key] = simulator.schedules
            simulator.finalize()
            del simulator
            
            if self.muted:
                sys.stdout = self.old_stdout

            return copy.deepcopy(self.cache[cache_key])

def main():
    """
    Run the simulator.
    """
    return mosaik_api.start_simulation(Simulator())

if __name__ == '__main__':
    main()

