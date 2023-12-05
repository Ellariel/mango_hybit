
"""
Gateway between mosaik and the MAS.

The class :class:`MosaikAPI` implements the `high-level mosaik API`.

The MosaikAPI also manages the root main_container for the MAS.  It starts
a :class:`MosaikAgent` and a :class:`mas.controller.Controller` agent within
that main_container.  The MosaikAgent serves as a gateway between the WecsAgents
and mosaik.  The Controller agent coordinates all WecsAgents of a wind farm.

The WecsAgents do not run within the root main_container but in separate containers
in sub processes.  These subprocesses are as well managed by the MosaikAPI.

The entry point for the MAS is the function :func:`main()`.  It parses the
command line arguments and starts the :func:`run()` coroutine which runs until
mosaik sends a *stop* message to the MAS.

.. _mosaik API:
   https://mosaik.readthedocs.org/en/latest/mosaik-api/low-level.html

"""

import asyncio
import logging

import mosaik_api

#from mango.core.agent import Agent
#from mango.core.container import Container
#from mango.messages.message import Performatives

from mango import create_container
from mango import Agent

#from src.mas.agent_messages import UpdateStateMessage, RequestInformationMessage, CurrentPMaxMessage, \
#    ControlActionsDone, TriggerControlActions, create_msg_content, read_msg_content
#from src.mas.controller import Controller
#from src.mas.wecs import WecsAgent

from src.mas.agent_messages import UpdateStateMessage, RegisterRequestMessage, \
    RegisterConfirmMessage, TriggerControlActions, UpdateConfirmMessage, \
ControlActionsDone, RequestInformationMessage, AnswerInformationMessage, create_msg_content, read_msg_content

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple




class CellAgent(Agent):
    def __init__(self, container, controller, initial_state={}):
        super().__init__(container)
        self.container = container
        self.controller = controller
        self.registration_confirmed = asyncio.Future()
        #print(f"Hello world! I am CellAgent. My id is {self.aid}, controller: {self.controller}")
        self.current_state = initial_state
        self.previous_states = []

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e. g. performative, sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        print(f"{self.aid} received: {content}")
        if isinstance(content, RegisterConfirmMessage):
            # This should be a confirm message by the controller that it has received our registration
            self.registration_confirmed.set_result(True)
        elif isinstance(content, UpdateStateMessage):
            # This should inform agents about the state of connected mosaik Entities
            self.update_state(content, meta)
        elif isinstance(content, RequestInformationMessage):
            # This should send information requested to the controller
            self.send_info(content, meta)
        else:
            pass

    def send_info(self, content, meta):
        """
        Request information of the current state or any calculated data of connected Entity
        :param state_msg: The state message including the information requested
        :param meta: the meta dict
        """

        data = {}
        for v in self.current_state.keys():
            values = [j for i in self.previous_states + [self.current_state] for j in i[v].values()]     
            data[v] = {'min' : min(values),
                       'max' : max(values),
                       'cur' : values[-1]}  

        # send info if sender_id is provided
        msg_content = create_msg_content(AnswerInformationMessage, info=data)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))        

    def update_state(self, content, meta):
        """
        Update the current state of connected Entity
        :param state_msg: The state message including the current state coming from mosaik
        :param meta: the meta dict
        """
        if len(self.current_state):
            self.previous_states.append(self.current_state) # store old state
        self.current_state = content.state # update current state
        # confirm if sender_id is provided
        msg_content = create_msg_content(UpdateConfirmMessage)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))

    async def register(self):
        """
        Schedule a task that sends a RegisterMessage to the ControllerAgent and wait for the confirmation.
        This should be called once an agent is initialized
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
        await asyncio.wait_for(self.registration_confirmed, timeout=3)

class ControllerAgent(Agent):
    def __init__(self, container, initial_state={}):
        super().__init__(container)
        self.container = container
        self.connected_agents = []
        #print(f"Hello world! I am ControllerAgent. My id is {self.aid}")
        self.current_state = initial_state
        self.previous_states = []
        self.requested_info = {}

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e. g. performative, sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        print(f"{self.aid} received: {content}")
        if isinstance(content, RegisterRequestMessage):
            self.handle_register_msg(content, meta)
        elif isinstance(content, UpdateStateMessage):
            self.update_state(content, meta)
        elif isinstance(content, TriggerControlActions):
            self.schedule_instant_task(self.perform_control_actions(meta))
        elif isinstance(content, RequestInformationMessage):
            # This should send information requested to the controller
            self.send_info(content, meta)
        elif isinstance(content, AnswerInformationMessage):
            self.requested_info[meta['conversation_id']].set_result(content.info)
        else:
            pass

    def send_info(self, content, meta):
        """
        Request information of the current state or any calculated data of connected Entity
        :param state_msg: The state message including the information requested
        :param meta: the meta dict
        """

        #data = {}
        #for v in self.current_state.keys():
        #    values = [j for i in self.previous_states + [self.current_state] for j in i[v].values()]     
        #    data[v] = {'min' : min(values),
        #               'max' : max(values),
        #               'cur' : values[-1]}  
                    
        # send info if sender_id is provided
        msg_content = create_msg_content(AnswerInformationMessage, info=data)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            )) 

    async def perform_control_actions(self, meta):
        """
        """
        #async def request_info():
            # Request information
        print('REQUEST')
        self.requested_info = {aid: asyncio.Future() for _, aid in self.connected_agents}
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                        receiver_addr=addr,
                        receiver_id=aid,
                        content=create_msg_content(RequestInformationMessage, info={'A':self.aid}),
                        acl_metadata={'conversation_id': aid,
                                    'sender_id': self.aid},
                ))
                for addr, aid in self.connected_agents]
        await asyncio.gather(*futs)
        # wait for info
        await asyncio.gather(*[fut for fut in self.requested_info.values()])
        print('self.requested_info', self.requested_info)
        

        #self.schedule_instant_task(request_info())
        print('ACTION')
        #print('self._info_confirm', self._info_confirm)

        # confirm if sender_id is provided
        msg_content = create_msg_content(ControlActionsDone)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))  

    def update_state(self, content, meta):
        """
        Update the current state of connected Entity
        :param state_msg: The state message including the current state coming from mosaik
        :param meta: the meta dict
        """
        if len(self.current_state):
            self.previous_states.append(self.current_state) # store old state
        self.current_state = content.state # update current state
        # confirm if sender_id is provided
        msg_content = create_msg_content(UpdateConfirmMessage)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))

    def handle_register_msg(self, content: RegisterRequestMessage, meta):
        """
        Registered an agent
        :param content: Register message
        :param meta information
        """
        addr = (content.host, content.port)
        aid = content.aid
        self.connected_agents.append((addr, aid))
        # reply with a confirmation
        msg_content = create_msg_content(RegisterConfirmMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(self.container.send_acl_message(
            receiver_addr=addr,
            receiver_id=aid,
            content=msg_content,
            acl_metadata={},
        ))

class MosaikAgent(Agent):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        # We need this to make sure all agents have received their updates and confirmations.
        self._updates_received = {}
        self._controllers_done = {}
        self._all_agents = {}
        self._controllers = {}
        print(f"Hello world! I am MosaikAgent. My id is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e. g. performative, sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        print(f"{self.aid} received: {content}")
        if isinstance(content, UpdateConfirmMessage):
            self._updates_received[meta['conversation_id']].set_result(True)
        elif isinstance(content, ControlActionsDone):
            self._controllers_done[meta['conversation_id']].set_result(True)
        else:
            pass

    def _reset(self):
        self._updates_received = {aid: asyncio.Future() for aid in self._all_agents.keys()}
        self._controllers_done = {aid: asyncio.Future() for aid in self._controllers.keys()}

    async def _update_agents(self, data):
        """
        Update the agents with new data from mosaik.
        """
        #print('update_agents')
        #print(data)
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=self._all_agents[mosaik_eid][0],
                    receiver_id=self._all_agents[mosaik_eid][1],
                    content=create_msg_content(UpdateStateMessage, state=input_data),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        # wait for confirmation
        await asyncio.gather(*[fut for fut in self._updates_received.values()])

    async def _trigger_control_cycle(self):
        """
        Trigger control cycle after all the agents got their updates from Mosaik.
        """
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=controller[0],
                    receiver_id=controller[1],
                    content=create_msg_content(TriggerControlActions),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, controller in self._controllers.items()]
        await asyncio.gather(*futs)
        # wait for confirmation
        await asyncio.gather(*[fut for fut in self._controllers_done.values()])

    async def loop_step(self, inputs):
            """
            This will be called from the mosaik api once per step.

            :param inputs: the input dict from mosaik: {eid_1: {'P': p_value}, eid_2: {'P': p_value}...}
            :return: the output dict: {eid_1: p_max_value, eid_2: p_max_value}
            """
            # 1. reset
            self._reset()
            # 2. update state of agents
            await self._update_agents(inputs)
            # 3. trigger control actions
            await self._trigger_control_cycle()

            return {}


#logger = logging.getLogger('mas.mosaik')


def main():
    """Run the multi-agent system."""
    logging.basicConfig(level=logging.INFO)
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
        self.cell_agents = {}  # eid : ((host,port), aid)
        self.controllers = {}  # eid : ((host,port), aid)
        # Set/updated in "setup_done()"
        self.all_agents = {} # contains agents + controllers for technical tasks
        self.entities = {}  # agent_id: unit_id

    def init(self, sid, time_resolution=1., **sim_params):
        self.sid = sid
        self.loop = asyncio.get_event_loop()
        self.main_container = self.loop.run_until_complete(self._create_container(self.host, self.port))
        self.mosaik_agent = self.loop.run_until_complete(self._create_mosaik_agent(self.main_container))
        return META

    async def _create_container(self, host, port):
        return await create_container(addr=(host, port))
    
    async def _create_mosaik_agent(self, container):
        return MosaikAgent(container)

    async def _create_agent(self, container, controller, initial_state={}):
        if controller == None:
            agent = ControllerAgent(container, initial_state)
        else:
            agent = CellAgent(container, self.controllers[controller], initial_state)
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
        if model_conf['controller'] == None:
            n_agents = len(self.controllers)
            for i in range(n_agents, n_agents + num):
                eid = 'ControllerAgent_%s' % i
                agent = self.loop.run_until_complete(self._create_agent(self.main_container, 
                                                                        None, 
                                                                        model_conf['initial_state'] ))
                self.controllers[eid] = (self.main_container.addr, agent.aid)
                entities.append({'eid': eid, 'type': model})
        else:
            n_agents = len(self.cell_agents)
            for i in range(n_agents, n_agents + num):
                eid = 'Agent_%s' % i
                agent = self.loop.run_until_complete(self._create_agent(self.main_container, 
                                                                        model_conf['controller'], 
                                                                        model_conf['initial_state']))
                self.cell_agents[eid] = (self.main_container.addr, agent.aid)
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
        self.all_agents = self.cell_agents.copy()
        self.all_agents.update(self.controllers)
        self.mosaik_agent._all_agents = self.all_agents
        self.mosaik_agent._controllers = self.controllers

        full_ids = ['%s.%s' % (self.sid, eid) for eid in self.all_agents.keys()]
        relations = yield self.mosaik.get_related_entities(full_ids)
        for full_aid, units in relations.items():
            # We should be connected to at least one entity
            assert len(units) >= 1
            uid, _ = units.popitem()
            # Create a mapping "agent ID -> unit ID"
            aid = full_aid.split('.')[-1]
            self.entities[aid] = uid
        #print(self.entities)

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
        # trigger the loop to enable agents to send / receive messages via run_until_complete
        output_dict = self.loop.run_until_complete(self.mosaik_agent.loop_step(inputs=inputs))

        # Make "set_data()" call back to mosaik to send the set-points:
        inputs = {aid: {self.entities[aid]: {'P_max': P_max}}
                  for aid, P_max in output_dict.items()}
        yield self.mosaik.set_data(inputs)

        return time + self.step_size
    
    def get_data(self, outputs):
        # we are going to send the data asynchronously via set_data, hence we do not need to implement get_data()
        pass

if __name__ == '__main__':
    main()


