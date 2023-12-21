
"""


"""

import asyncio
#import logging
import mosaik_api
#import mosaik_api_v3 as mosaik_api
import mango
import copy
from agent_messages import *
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# to run asyncio loop with updated mosaik
import nest_asyncio
nest_asyncio.apply()

from utils import highlight, reduce_zero_dict, reduce_equal_dicts
from utils import *

STATE_DICT = {
    'production' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
    },
    'consumption' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
    },    
}

class Agent(mango.Agent):
    def __init__(self, container, **kwargs):
        super().__init__(container)
        self.container = container
        self.controller = kwargs.get('controller', None)
        self.state = kwargs.get('initial_state', copy.deepcopy(STATE_DICT))
        self.connected_agents = []
        self._registration_confirmed = asyncio.Future()
        self._instructions_confirmed = {}
        self._requested_states = {}
        self._aggregated_state = {}

        self.verbose = kwargs.get('verbose', 2)
        self.input_method = kwargs.get('input_method', None)
        self.output_method = kwargs.get('output_method', None)
        if self.verbose >= 2:
            print(f"Hello world! I am a mango agent. My aid is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """

        content = read_msg_content(content)
        if self.verbose >= 2:
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
        #elif isinstance(content, TriggerCommunications):
        #    self.schedule_instant_task(self.perform_communications(meta))
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

        if callable(self.input_method):
            self.state = self.input_method(content.state, self.state)
        else:
            self.state = copy.deepcopy(content.state)

        # state={'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0}}
        '''
        for eid, value in content.state['current'].items():
            if 'Load' in eid: # check consumtion/production
                self.state['consumption']['current'] = abs(value)
            elif 'Gen' in eid or 'Wecs' in eid:
                self.state['production']['current'] = abs(value)
            elif 'Grid' in eid:
                if value >= 0:
                    self.state['production']['current'] = value
                    self.state['consumption']['current'] = 0
                else:
                    self.state['production']['current'] = 0
                    self.state['consumption']['current'] = value
        '''

        msg_content = create_msg_content(UpdateConfirmMessage)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))

    async def send_state(self, content, meta):
        """
        Send requested information (e.g. of the current state or any calculated data of connected Entity)
        :param content: The content including the information requested
        :param meta: The meta information dict
        """

        self._aggregated_state = copy.deepcopy(self.state)
               
        # Request information
        if len(self.connected_agents):
            if self.verbose >= 1:
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
            self._aggregated_state = get_aggregated_state(self._requested_states, self._aggregated_state)

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
        instructions = content.instructions
        instruction = instructions[self.aid]

        # Do execute instruction
        
        if self.verbose >= 1:
            print(f"{highlight(self.aid)} <- {highlight('current_state')}: {self._aggregated_state}, {highlight('new_state')}: {reduce_equal_dicts(instruction, self._aggregated_state)}")
        if len(self.connected_agents):
            delta = calc_delta(self._aggregated_state, instruction)
            additional_instructions, delta_remained = compose_instructions(self._requested_states, delta)
            instructions.update(additional_instructions)
            if self.verbose >= 1:
                print(f"{highlight('delta for')} {', '.join(self._requested_states.keys())}: {reduce_zero_dict(delta)}")
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

        msg_content = create_msg_content(InstructionsConfirmMessage, instructions={'aid': self.aid,
                                                                                   'instructions': content.instructions})
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            )) 

class MosaikAgent(mango.Agent):
    def __init__(self, container, **kwargs):
        super().__init__(container)
        self.container = container
        self.connected_agents = []
        self.state = kwargs.get('initial_state', copy.deepcopy(STATE_DICT))

        self._updates_received = {}
        #self._communications_done = {}
        self._instructions_confirmed = {}
        self._requested_states = {}
        self._aggregated_state = {}
        self._all_agents = {}
        #self._controllers = {}
        self.verbose = kwargs.get('verbose', 2)
        self.input_method = kwargs.get('input_method', None)
        self.output_method = kwargs.get('output_method', None)
        if self.verbose >= 2:
            print(f"Hello world! I am MosaikAgent. My aid is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        if self.verbose >= 2:
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
        if self.verbose >= 1:
            print(f"BROADCAST UPDATES: {', '.join([f'{eid}->{self._all_agents[eid][1]}' for eid in data.keys()])}")
        self._updates_received = {eid: asyncio.Future() for eid in data.keys()}
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=self._all_agents[mosaik_eid][0],
                    receiver_id=self._all_agents[mosaik_eid][1],
                    content=create_msg_content(UpdateStateMessage, state=input_data),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        await asyncio.gather(*[fut for fut in self._updates_received.values()])
        if self.verbose >= 1:
            print("UPDATES ARE CONFIRMED")

    async def trigger_communication_cycle(self):
        """
        Trigger control cycle after all the agents got their updates from Mosaik.
        """

        self._aggregated_state = copy.deepcopy(self.state)

        # Request information if there are connected agents
        if len(self.connected_agents):
            if self.verbose >= 1:
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
            self._aggregated_state = get_aggregated_state(self._requested_states, self._aggregated_state)
            if self.verbose >= 1:
                print(f"STOP COMMUNICATION CYCLE: {', '.join([k for k in self._requested_states.keys()])}")
                print('EXECUTE REDISPATCH ALGORITHM')
        # grid state: {'production': {'min': 0, 'max': 0, 'current': 1.0040492208704848}, 'consumption': {'min': 0, 'max': 0, 'current': 0}}
        # requested_info: {'agent2': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 1.0}}, 
        #                  'agent3': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 1.0}}}
        # aggregated data: {'production': {'min': 0, 'max': 0, 'current': 2.0}, 'consumption': {'min': 0, 'max': 0, 'current': 2.0}}


            print(self.state)
            grid_delta, cell_delta = compute_delta_state(self.state, self._aggregated_state)
            #new_grid_state = add_delta(self.state, grid_delta)
            instructions, feedback = compose_instructions(self._requested_states, cell_delta)
            #instructions.update({self.aid : new_grid_state, 'delta' : cell_delta})

            # Send instructions
            if self.verbose >= 1:
                print('BROADCAST REDISPATCH INSTRUCTIONS')
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
            if self.verbose >= 1:
                print('INSTRUCTIONS ARE CONFIRMED')


        output_state =  {}#{eid : info['instructions'] for eid, a in self._all_agents.items() 
                           #                     for info in self._instructions_confirmed[0] 
                           #                         if a[1] == info['aid']}
        if callable(self.output_method):
            return self.output_method(output_state)
        return copy.deepcopy(output_state)        

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

#logger = logging.getLogger('mosaik_agents')
def main():
    """Run the multi-agent system."""
    #logging.basicConfig(level=logging.INFO)
    return mosaik_api.start_simulation(MosaikAgents())

# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'MosaikAgents': {
            'public': True,
            'any_inputs': True,
            'params': ['controller', 'initial_state'],
            'attrs': [],
        },
    },
}

class MosaikAgents(mosaik_api.Simulator):
    """
    Interface to mosaik.
    """

    def __init__(self):
        super().__init__(META)
        self.step_size = 60 * 15  # We have a step size of 15 minutes specified in seconds:
        self.host = 'localhost'
        self.port = 5678
        # Set by "run()":
        self.mosaik = None  # Proxy object for mosaik
        # Set in "init()"
        self.sid = None  # Mosaik simulator ID
        self.loop = None  # Mango agents loop
        self.main_container = None # Mango container
        self.mosaik_agent = None # Mosaik Mango Agent
        # Updated in "create()"
        #self.cell_agents = {}  # eid : ((host,port), aid)
        self.controllers = {}  # eid : ((host,port), aid)
        # Set/updated in "setup_done()"
        self.all_agents = {} # contains agents + controllers for technical tasks
        self.entities = {}  # agent_id: unit_id

        self.verbose = 2
        self.input_method = None
        self.output_method = None

    def init(self, sid, time_resolution=1., **sim_params):
        self.sid = sid
        self.loop = asyncio.get_event_loop()

        self.verbose = sim_params.get('verbose', self.verbose)
        self.input_method = sim_params.get('input_method', self.input_method)
        self.output_method = sim_params.get('output_method', self.output_method)

        self.main_container = self.loop.run_until_complete(self._create_container(self.host, self.port))
        self.mosaik_agent = self.loop.run_until_complete(self._create_mosaik_agent(self.main_container))
        return META

    async def _create_container(self, host, port):
        return await mango.create_container(addr=(host, port))
    
    async def _create_mosaik_agent(self, container):
        return MosaikAgent(container, verbose=self.verbose)

    async def _create_agent(self, container, **kwargs):
        controller = kwargs.get('controller', None)
        agent = Agent(container, 
                      controller=(self.mosaik_agent.container.addr, self.mosaik_agent.aid) if controller == None else self.all_agents[controller], 
                      verbose=self.verbose, 
                      input_method=self.input_method,
                      output_method=self.output_method)
        await agent.register()
        return agent

    def create(self, num, model, **model_conf):
        """
        Create *num* instances of *model* and return a list of entity dicts
        to mosaik.
        """
        assert model in META['models']
        # Get the number of agents created so far and count from this number
        # when creating new entity IDs:
        entities = []
        n_agents = len(self.all_agents) + 1
        
        for i in range(n_agents, n_agents + num):
                eid = 'Agent_%s' % i
                agent = self.loop.run_until_complete(self._create_agent(self.main_container, **model_conf))
                self.all_agents[eid] = (self.main_container.addr, agent.aid)
                #if controller == None:
                #    self.controllers[eid] = (self.main_container.addr, agent.aid)
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
        #self.mosaik_agent._controllers = self.controllers
        full_ids = ['%s.%s' % (self.sid, eid) for eid in self.all_agents.keys()]
        relations = yield self.mosaik.get_related_entities(full_ids)
        print('relations:', relations)
        for full_aid, units in relations.items():
            if len(units):
                # We should be connected to at least one entity
                #assert len(units) >= 1
                uid, _ = units.popitem()
                # Create a mapping "agent ID -> unit ID"
                aid = full_aid.split('.')[-1]
                self.entities[aid] = uid
        print('entities:', self.entities)

    def finalize(self):
        self.loop.run_until_complete(self._shutdown(self.main_container))

    @staticmethod
    async def _shutdown(*args):
        futs = []
        for arg in args:
            futs.append(arg.shutdown())
        print('Going to shutdown agents and container... ', end='')
        await asyncio.gather(*futs)
        print('done.')

    def step(self, time, inputs, max_advance):
        """Send the inputs of the controlled unites to our agents and get new
        set-points for these units from the agents.

        This method will run for at most "step_size" seconds, even if the
        agents need longer to do their calculations.  They will then continue
        to do stuff in the background, while this method returns and allows
        mosaik to continue the simulation.

        """
        print('\ninputs:', inputs)
        # trigger the loop to enable agents to send / receive messages via run_until_complete
        output_dict = self.loop.run_until_complete(self.mosaik_agent.run_loop(inputs=inputs))


        print('\noutput', output_dict)
        #print(self.all_agents)

        # Make "set_data()" call back to mosaik to send the set-points:
        #inputs = {aid: {self.entities[aid]: v}
        #          for aid, v in output_dict.items()}
        
        inputs = {}
        
        yield self.mosaik.set_data(inputs)

        return time + self.step_size
    
    def get_data(self, outputs):
        # we are going to send the data asynchronously via set_data, hence we do not need to implement get_data()
        pass

if __name__ == '__main__':
    main()


