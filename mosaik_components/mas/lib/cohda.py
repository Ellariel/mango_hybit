import logging
import asyncio
import mosaik_api_v3 as mosaik_api
from mosaik_cohda.des_flex.flex_class import Flexibility
from mosaik_cohda.start_values import StartValues
from mosaik_cohda.cohda_simulator import CohdaSimulator
from mango.core.agent import Agent
from mango.core.container import Container
from mango.role.core import RoleAgent
from mosaik_cohda.agent_roles import FlexReceiverRole, FlexCohdaRole, \
    FlexTerminationRole, FlexNegotiationStarterRole, TerminationData
from mosaik_cohda.start_values import StartValues, SolutionSchedule
from mosaik_cohda.mango_library.coalition.core import CoalitionParticipantRole
import nest_asyncio

class FlexReceiverRoleModified(FlexReceiverRole):
    def setup(self) -> None:
        super().setup()
        self.id = int(self.context.aid.split('agent')[1]) + 1

    def on_change_model(self, model) -> None:
        if model.terminated:
            schedule = self.context.data.current_cohda_solution.solution_candidate.candidate[self.id]
            solution = SolutionSchedule(schedule)
            self.context.schedule_instant_task(
                self.context.send_message(
                    receiver_addr=self._mosaik_agent_address[0],
                    receiver_id=self._mosaik_agent_address[1],
                    content=solution,
                    acl_metadata={'conversation_id':
                                      self._mosaik_agent_address[1],
                                  'sender_id': self.context.aid},
                    create_acl=True,
                )
            )

class Simulator(CohdaSimulator):

    async def create_flex_agent(self, container: Container,
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
            pending = asyncio.all_tasks()
            self.loop.run_until_complete(asyncio.gather(*pending))
            self.loop.run_until_complete(
                self._shutdown(self.main_container, 
                               self.agent_container, 
                               self.mosaik_agent))
        except:
            pass
        
class COHDA():
    def __init__(self, step_size=15*60, time_resolution=1., host='localhost', base_port=6060, **sim_params):
        self.step_size = step_size
        self.sim_params = sim_params
        self.time_resolution = time_resolution
        self.host = host
        self.base_port = base_port
        self.time_step = 0

    def reinitialize(self, n_agents):
        self.simulator = Simulator()
        self.simulator.host = self.host
        self.simulator.port = self.base_port
        self.base_port += n_agents
        self.simulator.init(sid=0, step_size=self.step_size, time_resolution=self.time_resolution, **self.sim_params)
        agent_params = {'control_id': 0, 
                'time_resolution': self.time_resolution,
                'step_size' : self.step_size}
        self.agents = []
        for i in range(n_agents):
            agent_model = self.simulator.create(1, 'FlexAgent', **agent_params)
            self.agents.append(agent_model)
        self.simulator.setup_done()

    def execute(self, target_schedule, flexibility):
            
            n_agents = len(flexibility)
            participants = list(range(n_agents))
            self.reinitialize(n_agents=n_agents)

            input_data = {}
            output_data = {}
            self.simulator.uids = {}
            for i in participants:
                agent, flex = self.agents[i], flexibility[i]
                eid = agent[0]['eid']
                data = {'StartValues': {'ID_0': StartValues(schedule=target_schedule, 
                                                            participants=participants)},
                        'Flexibility': {'ID_0': Flexibility(flex_max_power=flex['flex_max_power'],
                                                            flex_min_power=flex['flex_min_power'],
                                                            flex_max_energy_delta=[1] * len(flex['flex_min_power']),
                                                            flex_min_energy_delta=[0] * len(flex['flex_min_power']),
                                                        )}
                }
                input_data[eid] = data
                self.simulator.uids[eid] = None
                output_data[eid] = ['FlexSchedules'] 

            self.simulator.step(time=self.time_step, inputs=input_data, max_advance=0)
            self.time_step += 1
            self.simulator.get_data(outputs=output_data)
            output_data = self.simulator.schedules
            self.simulator.finalize()
            del self.simulator

            return output_data

def main():
    """
    Run the simulator.
    """
    nest_asyncio.apply()
    logging.disable()
    return mosaik_api.start_simulation(Simulator())

if __name__ == '__main__':
    main()

