
"""


"""

import asyncio
import mango
import copy
import time
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import mosaik_api_v3 as mosaik_api
from mosaik_components.mas.agent_messages import *
from mosaik_components.mas.utils import *

# to run asyncio loop with updated mosaik
import nest_asyncio
nest_asyncio.apply()

# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'event-based',
    'models': {
        'MosaikAgents': {
            'public': True,
            'any_inputs': True,
            'params': ['controller', 'initial_state'],
            'attrs': ['steptime'],
        },
    },
}

class Agent(mango.Agent):
    def __init__(self, container, **params):
        super().__init__(container)
        self.container = container
        self.params = params
        self.state = self.params.pop('initial_state', copy.deepcopy(self.params['state_dict']))
        self.controller = self.params.pop('controller', None)
        self.sid = self.params.pop('sid', None)
        self.aeid = self.params.pop('aeid', None)
        #print(self.sid, self.aeid, self.aid)
        self.connected_agents = []
        self._registration_confirmed = asyncio.Future()
        self._instructions_confirmed = {}
        self._requested_states = {}
        self._aggregated_state = {}
        self.current_time = -1
        self.first_time_step = True
        if self.params['verbose'] >= 2:
            print(f"Hello world! I am a mango agent. My aid is {self.aid}.{' I am a Controller!' if self.controller[1] == 'agent0' else ''}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """

        content = read_msg_content(content)
        if self.params['verbose'] >= 2:
            print(f"{self.aid} received: {content}")
        if isinstance(content, RegisterRequestMessage):
            self.register_agent(content, meta)
        elif isinstance(content, RegisterConfirmMessage):
            self._registration_confirmed.set_result(True)
        elif isinstance(content, UpdateStateMessage):
            self.update_state(content, meta)
        elif isinstance(content, RequestStateMessage):
            self.schedule_instant_task(self.send_state(content, meta))
        elif isinstance(content, AnswerStateMessage):
            self._requested_states[meta['conversation_id']].set_result(content.state)
        elif isinstance(content, BroadcastInstructionsMessage):
            self.schedule_instant_task(self.execute_instructions(content, meta))
        elif isinstance(content, InstructionsConfirmMessage):
            self._instructions_confirmed[meta['conversation_id']].set_result(content.instructions)
        else:
            pass

    async def register(self):
        """
        Schedule a task that sends a RegisterRequestMessage to the controller agent and wait for the confirmation.
        This should be called once an agent is initialized.
        """
        msg_content = create_msg_content(RegisterRequestMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(
            coroutine=self.container.send_acl_message(
                content=msg_content,
                receiver_addr=self.controller[0], 
                receiver_id=self.controller[1],
                acl_metadata={}
            ))
        await asyncio.wait_for(self._registration_confirmed, timeout=3)

    def register_agent(self, content: RegisterRequestMessage, meta):
        """
        Registered an agent
        :param content: Register message
        :param meta: The meta information dict
        """
        addr = (content.host, content.port)
        aid = content.aid
        self.connected_agents.append((addr, aid))
        msg_content = create_msg_content(RegisterConfirmMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(self.container.send_acl_message(
            receiver_addr=addr,
            receiver_id=aid,
            content=msg_content,
            acl_metadata={},
        ))

    def update_state(self, content, meta):
        """
        Update the current state of connected Entity
        :param content: The state content including the current state coming from mosaik
        :param meta: the meta information dict
        """
        
        info = content.state.pop('MosaikAgent', {})
        self.current_time = info['current_time']
        self.first_time_step = info['first_time_step']

        if callable(self.params['input_method']):
            self.state = self.params['input_method'](self.aeid, 
                                                     self.aid, 
                                                     content.state, 
                                                     self.state,
                                                     self.current_time,
                                                     self.first_time_step,
                                                     **self.params)
        else:
            self.state = copy.deepcopy(content.state)

        msg_content = create_msg_content(UpdateConfirmMessage)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))

    def aggregate_states(self, requested_states, current_state=None):
        if callable(self.params['states_agg_method']):
            return self.params['states_agg_method'](self.aeid, self.aid, requested_states, current_state)

    async def send_state(self, content, meta):
        """
        Send requested information (e.g. of the current state or any calculated data of connected Entity)
        :param content: The content including the information requested
        :param meta: The meta information dict
        """

        self._aggregated_state = copy.deepcopy(self.state)
               
        # Request information
        if len(self.connected_agents):
            if self.params['verbose'] >= 1:
                print(f"REQUEST STATES: {self.aid} <- {', '.join([i[1] for i in self.connected_agents])}")

            self._requested_states = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(RequestStateMessage, state={}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._requested_states = await asyncio.gather(*[fut for fut in self._requested_states.values()])
            self._requested_states = {k : v for i in self._requested_states for k, v in i.items()}
            # {'agent4': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 1.0}}})
            self._aggregated_state = self.aggregate_states(self._requested_states, self._aggregated_state)

        msg_content = create_msg_content(AnswerStateMessage, state={self.aid : self._aggregated_state})
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))        

    async def execute_instructions(self, content, meta):
        """
        Execute received instructions and sent the confirmation
        :param content: The content including instructions to execute
        :param meta: The meta information dict
        """
        ok = False
        instructions = content.instructions
        instruction = instructions[self.aid]

        if callable(self.params['execute_method']):
            #print(self.params)
            ok, add_instructions, state = self.params['execute_method'](self.aeid,
                                                                   self.aid, instruction=instruction, 
                                                                   current_state=self._aggregated_state,
                                                           requested_states=self._requested_states,
                                                           current_time=self.current_time,
                                                           first_time_step=self.first_time_step)

            self.state = state
            instructions.update(add_instructions)
        else:
            raise AttributeError('Redispatch method is not defined!')

        if self.params['verbose'] >= 1:
            print(f"{highlight(self.aid)} <- {highlight('current_state')}: {self._aggregated_state}, {highlight('new_state')}: {reduce_equal_dicts(self.state, self._aggregated_state)}")
        
        self._instructions_confirmed = instructions

        if len(self.connected_agents):
            # Broadcast instructions
            self._instructions_confirmed = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(BroadcastInstructionsMessage, instructions=instructions),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._instructions_confirmed = await asyncio.gather(*[fut for fut in self._instructions_confirmed.values()])
            self._instructions_confirmed = {k : v for i in self._instructions_confirmed for k, v in i['instructions'].items()}

        if ok:        
            msg_content = create_msg_content(InstructionsConfirmMessage, instructions={'aid': self.aid,
                                                                                    'instructions': self._instructions_confirmed})
            if 'sender_id' in meta.keys():
                conv_id = meta.get('conversation_id', None)
                self.schedule_instant_task(self.container.send_acl_message(
                    content=msg_content,
                    receiver_addr=meta['sender_addr'],
                    receiver_id=meta['sender_id'],
                    acl_metadata={'conversation_id': conv_id},
                ))
        else:
            raise AttributeError('Redispatch method did not converged!')


class MosaikAgent(mango.Agent):
    def __init__(self, container, **params):
        super().__init__(container)
        self.container = container
        self.params = params
        self.sid = self.params.pop('sid', None)
        self.aeid = self.params.pop('aeid', None)

        self.converged = 0
        self.convergence_steps = self.params.pop('convergence_steps', 2)
        self.convegence_max_steps = self.params.pop('convegence_max_steps', 5)

        self.cached_solution = {}
        self.current_time = -1
        self.first_time_step = True

        self.state = self.params.pop('initial_state', copy.deepcopy(self.params['state_dict']))
        self.connected_agents = []
        self._updates_received = {}
        self._instructions_confirmed = {}
        self._requested_states = {}
        self._aggregated_state = {}
        self._all_agents = {}
        if self.params['verbose'] >= 2:
            print(f"Hello world! I am MosaikAgent. My aid is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        if self.params['verbose'] >= 2:
            print(f"{self.aid} received: {content}")
        if isinstance(content, RegisterRequestMessage):
            self.register_agent(content, meta)
        elif isinstance(content, UpdateConfirmMessage):
            self._updates_received[meta['conversation_id']].set_result(True)
        elif isinstance(content, AnswerStateMessage):
            self._requested_states[meta['conversation_id']].set_result(content.state)
        elif isinstance(content, InstructionsConfirmMessage):
            self._instructions_confirmed[meta['conversation_id']].set_result(content.instructions)
        else:
            pass

    def register_agent(self, content: RegisterRequestMessage, meta):
        """
        Registered an agent
        :param content: Register message
        :param meta: The meta information dict
        """
        addr = (content.host, content.port)
        aid = content.aid
        self.connected_agents.append((addr, aid))
        msg_content = create_msg_content(RegisterConfirmMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(self.container.send_acl_message(
            receiver_addr=addr,
            receiver_id=aid,
            content=msg_content,
            acl_metadata={},
        ))

    async def update_agents(self, data):
        """
        Update the agents with new data from mosaik.
        """
        info = {'MosaikAgent': {
                        'current_time' : self.current_time,
                        'first_time_step' : self.first_time_step,
                    }}
        if self.params['verbose'] >= 1:
            print(f"BROADCAST UPDATES: {', '.join([f'{eid}->{self._all_agents[eid][1]}' for eid in data.keys()])}")
        self._updates_received = {eid: asyncio.Future() for eid in data.keys()}
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=self._all_agents[mosaik_eid][0],
                    receiver_id=self._all_agents[mosaik_eid][1],
                    content=create_msg_content(UpdateStateMessage, state={**input_data, **info}),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        await asyncio.gather(*[fut for fut in self._updates_received.values()])
        if self.params['verbose'] >= 1:
            print("UPDATES ARE CONFIRMED")

    def aggregate_states(self, requested_states, current_state=None):
        if callable(self.params['states_agg_method']):
            return self.params['states_agg_method'](self.aeid, self.aid, requested_states, current_state)
        else:
            raise AttributeError('States aggregation method is not defined!')

    async def trigger_communication_cycle(self):
        """
        Trigger communication cycle after all the agents got their updates from Mosaik.
        """
        # Request information if there are connected agents
        if len(self.connected_agents):
            if self.params['verbose'] >= 1:
                print(f"START COMMUNICATION CYCLE: {', '.join([aid for _, aid in self.connected_agents])}")
            self._requested_states = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(RequestStateMessage, state={}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._requested_states = await asyncio.gather(*[fut for fut in self._requested_states.values()])
            self._requested_states = {k : v for i in self._requested_states for k, v in i.items()}
            if self.params['verbose'] >= 1:
                print(f"STOP COMMUNICATION CYCLE: {', '.join([k for k in self._requested_states.keys()])}")
                print('EXECUTE REDISPATCH ALGORITHM')
                
            if callable(self.params['execute_method']):
                if self.converged < self.convergence_steps:
                    ok, self.cached_solution, state = self.params['execute_method'](self.aeid,
                                                                    self.aid, instruction=None, 
                                                                    current_state=self.state,
                                                            requested_states=self._requested_states,
                                                            current_time=self.current_time,
                                                            first_time_step=self.first_time_step,
                                                            **self.params)
                    self.state = state
                    self.converged += (1 if ok else 0)
                    if self.converged >= self.convergence_steps:
                        if self.params['verbose'] >= 1:
                            print(highlight('CONVERGED!', 'green'))                  
            else:
                raise AttributeError('Redispatch method is not defined!')
            
            if self.params['verbose'] >= 1:
                print('AGGREGATED INSTRUCTIONS:', self.cached_solution)
            
            # Send instructions
            if self.params['verbose'] >= 1:
                print(f'BROADCAST REDISPATCH INSTRUCTIONS')

            self._instructions_confirmed = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(BroadcastInstructionsMessage, instructions=self.cached_solution),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)

            self._instructions_confirmed = await asyncio.gather(*[fut for fut in self._instructions_confirmed.values()])
            self._instructions_confirmed = {k : v for i in self._instructions_confirmed for k, v in i['instructions'].items()}
            if self.params['verbose'] >= 1:
                print('INSTRUCTIONS ARE CONFIRMED')

            return self._instructions_confirmed   

    async def run_loop(self, inputs):
            """
            This will be called from the mosaik api once per step.

            :param inputs: the input dict from mosaik: {eid_1: {'P': p_value}, eid_2: {'P': p_value}...}
            :return: the output dict: {eid_1: p_max_value, eid_2: p_max_value}
            """
            # 1. update state of agents
            await self.update_agents(inputs)
            # 2. trigger control actions
            return await self.trigger_communication_cycle()

class MosaikAgents(mosaik_api.Simulator):
    """
    Interface to mosaik.
    """

    def __init__(self):
        super().__init__(META)
        self.host = None
        self.port = None
        self.params = {}
        # Set by "run()":
        self.mosaik = None  # Proxy object for mosaik
        # Set in "init()"
        self.sid = None  # Mosaik simulator ID
        self.loop = None  # Mango agents loop
        self.main_container = None # Mango container
        self.mosaik_agent = None # Mosaik Mango Agent
        # Set/updated in "setup_done()"
        self.all_agents = {} # contains agents + controllers for technical tasks
        self.aid_to_eid = {}
        self.entities = {}  # agent_id: unit_id
        self.output_cache = {}
        self._steptime = [] # performance

    def init(self, sid, time_resolution=1., **sim_params):
        #print(sim_params)
        self.sid = sid
        self.loop = asyncio.get_event_loop()
        self.params = sim_params
        self._steptime = []
        self.step_size = self.params.pop('step_size', 60*15)
        self.host = self.params.pop('host', '0.0.0.0')
        self.port = self.params.pop('port', 5678)
        self.params.setdefault('verbose', 1)
        self.params.setdefault('performance', True)
        self.params.setdefault('convergence_steps', 2)
        self.params.setdefault('convegence_max_steps', 5)
        return sim_params.pop('META', META)

    async def _create_container(self, host, port, **params):
        return await mango.create_container(addr=(host, port))
    
    async def _create_mosaik_agent(self, container, **params):
        params.update({'aeid' : 'MosaikAgent'})
        return MosaikAgent(container, **params)

    async def _create_agent(self, container, **params):
        controller = params.pop('controller', None)
        agent = Agent(container, 
                      controller=(self.mosaik_agent.container.addr, self.mosaik_agent.aid) if controller == None else self.all_agents[controller],
                      **params)
        await agent.register()
        return agent

    def create(self, num, model, **model_conf):
        return self.loop.run_until_complete(self._create(num, model, **model_conf))

    async def _create(self, num, model, **model_conf): #**self.params
        """
        Create *num* instances of *model* and return a list of entity dicts
        to mosaik.
        """
        assert model in META['models']
        # Get the number of agents created so far and count from this number
        # when creating new entity IDs:

        model_conf.update(self.params)
        model_conf.update({'sid' : self.sid})
        if model == 'MosaikAgents' and self.main_container == None and self.mosaik_agent == None:
            self.main_container = await self._create_container(self.host, self.port, **model_conf)
            self.mosaik_agent = await self._create_mosaik_agent(self.main_container, **model_conf)
            return [{'eid': 'MosaikAgent', 'type': model}]

        entities = []
        
        n_agents = len(self.all_agents) + 1
        
        for i in range(n_agents, n_agents + num):
                eid = 'Agent_%s' % i
                model_conf.update({'aeid' : eid})
                agent = await self._create_agent(self.main_container, **model_conf)
                self.all_agents[eid] = (self.main_container.addr, agent.aid)
                entities.append({'eid': eid, 'type': model})                  
            # as the event loop is not running here, we have to create the agents via loop.run_unti_complete.
            # That means however, that the agents are not able to receive or send messages unless the loop is
            # triggered from this module.
        return entities

    def setup_done(self):
        """
        Get the entities that our agents are connected to once the scenario
        setup is done.
        """
        self.mosaik_agent._all_agents = self.all_agents
        self.aid_to_eid = {v[1] : k for k, v in self.all_agents.items()}
        self.aid_to_eid[self.mosaik_agent.aid] = 'MosaikAgent'

        full_ids = ['%s.%s' % (self.sid, eid) for eid in self.all_agents.keys()] + [f"{self.sid}.MosaikAgent"]
        relations = yield self.mosaik.get_related_entities(full_ids)
        if self.params['verbose'] >= 2:
            print('relations:', relations)
        for full_aid, units in relations.items():
            if len(units):
                # We should be connected to at least one entity
                #assert len(units) >= 1
                uid, _ = units.popitem()
                # Create a mapping "agent ID -> unit ID"
                aid = full_aid.split('.')[-1]
                self.entities[aid] = uid
        if self.params['verbose'] >= 2:
            print('entities:', self.entities)

        if callable(self.params['initialize']):
            self.params['initialize'](**self.params)

    def finalize(self):
        # pending = asyncio.all_tasks()
        # self.loop.run_until_complete(asyncio.gather(*pending))
        self.loop.run_until_complete(self._shutdown(self.main_container))

        if callable(self.params['finalize']):
            self.params['finalize'](**self.params)

    @staticmethod
    async def _shutdown(*args):
        futs = []
        for arg in args:
            futs.append(arg.shutdown())
        #print('Going to shutdown agents and container... ', end='')
        await asyncio.gather(*futs)
        #print('done.')

    def step(self, current_time, inputs, max_advance):
        """Send the inputs of the controlled unites to our agents and get new
        set-points for these units from the agents.

        This method will run for at most "step_size" seconds, even if the
        agents need longer to do their calculations.  They will then continue
        to do stuff in the background, while this method returns and allows
        mosaik to continue the simulation.

        """
        if self.mosaik_agent.current_time != current_time:
            if self.params['verbose'] >= 1:
                print(highlight('\nNEW TIMESTEP:', 'white'), current_time)
            self._steptime = [] 
            self.mosaik_agent.converged = 0 
            self.mosaik_agent.first_time_step = True      
        else:
            if self.params['verbose'] >= 1:
                print(highlight('\nTIMESTEP:', 'white'), current_time)  
        self.mosaik_agent.current_time = current_time

        if self.params['verbose'] >= 2:
            print(highlight('\nINPUT:'), inputs)

        if self.mosaik_agent.converged >= self.mosaik_agent.convergence_steps:
            return current_time + self.step_size

        new_state = inputs.pop('MosaikAgent', {})
        if callable(self.params['input_method']):
            self.mosaik_agent.state = self.params['input_method']('MosaikAgent', 
                                                                  'agent0', 
                                                                  new_state, 
                                                                  self.mosaik_agent.state,
                                                                  self.mosaik_agent.current_time,
                                                                  self.mosaik_agent.first_time_step,
                                                                  **self.params)
        else:
            self.mosaik_agent.state = copy.deepcopy(new_state)
  
        steptime = time.time()
        output = self.loop.run_until_complete(self.mosaik_agent.run_loop(inputs=inputs))
        steptime = time.time() - steptime
        output = {self.aid_to_eid[k]: v for k, v in output.items()}    

        if self.mosaik_agent.converged <= self.mosaik_agent.convergence_steps:
            self.output_cache = output
            self.output_cache["MosaikAgent"] = self.mosaik_agent.state
            self._steptime += [steptime]       
        else:
            self.output_cache["MosaikAgent"] = self.mosaik_agent.state

        return current_time + self.step_size

    def get_data(self, outputs):
        data = {}

        if callable(self.params['output_method']):
            for aeid, attrs in outputs.items():
                aid = self.all_agents[aeid][1] if aeid in self.all_agents else None
                if aeid == "MosaikAgent" and 'steptime' in attrs:
                    aid = 'agent0'
                    attrs = attrs.copy()
                    attrs.remove('steptime')

                data[aeid] = self.params['output_method'](aeid, aid,
                                                         attrs, 
                                                         self.output_cache[aeid],
                                                         bool(self.mosaik_agent.converged >= self.mosaik_agent.convergence_steps),
                                                         self.mosaik_agent.current_time,
                                                         self.mosaik_agent.first_time_step,
                                                         **self.params)
        else:
            data = self.output_cache

        if "MosaikAgent" in data and self.params['performance']:
            data["MosaikAgent"].update({'steptime' : sum(self._steptime)})

        if self.params['verbose'] >= 2:
            print(highlight('\nOUTPUT:'), data)

        self.mosaik_agent.first_time_step = False
        
        return data


def main():
    """Run the multi-agent system."""
    return mosaik_api.start_simulation(MosaikAgents())

if __name__ == '__main__':
    main()


