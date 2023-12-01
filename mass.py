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

from src.mas.agent_messages import UpdateStateMessage, RequestInformationMessage, CurrentPMaxMessage, \
    ControlActionsDone, TriggerControlActions, create_msg_content, read_msg_content
from src.mas.controller import Controller
from src.mas.wecs import WecsAgent

from src.mas.agent_messages import UpdateStateMessage, SetPMaxMessage, RequestInformationMessage, RegisterMessage, \
    CurrentPMessage, CurrentPMaxMessage, create_msg_content, read_msg_content

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger('mas.mosaik')


class MangoAgent(Agent):
    """
    A WecsAgent is the “brain” of a simulated or real WECS.
    """

    def __init__(self, container: Container, controller_addr, controller_id, model_conf: Dict):
        super().__init__(container)

        self.controller = (controller_addr, controller_id)
        self.model_conf = model_conf
        self.data = None
        self.p_max = None               # The maximum allowed generation, received by the controller
        self.current_p = None           # The power that the plant would currently generate
        self.registration_confirmed = asyncio.Future()

    @classmethod
    async def create(cls, container: Container, controller_addr: Tuple[str, int],
                     controller_id: str, model_conf: Dict):
        """Return a new :class:`WecsAgent` instance.

        *main_container* is the main_container that the agent lives in.

        *controller_address* is the address of the Controller agent that the
        agent will register with.

        *model_conf* is a dictionary containing values for *P_rated*,
        *v_rated*, *v_min*, *v_max* (see the WECs model for details).

        """
        # We use a factory function here because __init__ cannot be a coroutine
        # and we want to make sure the agent is registered at the controller and that
        # when we return the instance we have have a fully initialized
        # instance.
        #
        # Classmethods don't receive an instance "self" but the class object as
        # an argument, hence the argument name "cls".

        agent = cls(container=container, controller_addr=controller_addr, controller_id=controller_id,
                   model_conf=model_conf)
        await agent._register(controller_addr=controller_addr, controller_id=controller_id)
        return agent

    def handle_msg(self, content, meta: Dict[str, Any]):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e. g. performative, sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        performative = meta.get('performative', None)
        # We first filter the messages regarding its performative
        if performative == Performatives.inform:
            # inform messages may inform about current_p (from MosaikAgent) or about p_max (from ControllerAgent)
            if isinstance(content, UpdateStateMessage):
                self.update_state(content, meta)
            elif isinstance(content, SetPMaxMessage):
                self.set_p_max(content, meta)
            else:
                self.handle_unknown_message(content, meta)

        elif performative == Performatives.request:
            # request messages may ask for the current_p (from ControllerAgent) or for p_max (from MosaikAgent).
            # Both are handled in reply_to_information_request
            if isinstance(content, RequestInformationMessage):
                self.reply_to_information_request(content, meta)
            else:
                self.handle_unknown_message(content, meta)

        elif performative == Performatives.confirm:
            # This should be a confirm message by the controller that it has received our registration
            self.registration_confirmed.set_result(True)
        else:
            # We expect a performative to all incoming messages
            self.handle_unknown_message(content, meta)

    def update_state(self, state_msg: UpdateStateMessage, meta: Dict[str, any]):
        """
        Update the current state (current_p) of the simulated WECS
        :param state_msg: The state message including the current state coming from mosaik
        :param meta: the meta dict
        """
        self.current_p = state_msg.state['P']

        # confirm if sender_id is provided
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self._container.send_message(
                content=None,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'performative': Performatives.confirm, 'conversation_id': conv_id},
                create_acl=True
            ))

    async def _register(self, controller_addr, controller_id):
        """
        Schedule a task that sends a RegisterMessage to the ControllerAgent and wait for the confirmation.
        This should be called once an agent is initialized

        """
        msg_content = create_msg_content(RegisterMessage,
                                         aid=self.aid, host=self._container.addr[0], port=self._container.addr[1])
        self.schedule_instant_task(
            coroutine=self._container.send_message(
                content=msg_content,
                receiver_addr=controller_addr, receiver_id=controller_id,
                create_acl=True,
                acl_metadata={'performative': Performatives.request}
            ))
        await asyncio.wait_for(self.registration_confirmed, timeout=3)

    def set_p_max(self, p_max_msg: SetPMaxMessage, meta: Dict[str, any]):
        """
        Sets a new power limit *p_max* for the
        simulated wecs.
        :param p_max_msg: The SetPMaxMessage
        :param meta: the meta dict
        """
        p_max = p_max_msg.p_max
        if p_max is None:
            # if p_max is None, p_max can be the maximum power of the plant, which can be found in model_conf
            p_max = self.model_conf['P_rated']
        self.p_max = p_max

        # send a confirmation to the controller
        receiver_addr = meta.get('sender_addr', None)
        receiver_id = meta.get('sender_id', None)
        conversation_id = meta.get('conversation_id', None)
        self.schedule_instant_task(
            self._container.send_message(
                content=None, receiver_addr=receiver_addr,
                receiver_id=receiver_id, create_acl=True, acl_metadata={
                    'sender_id': self.aid,
                    'conversation_id': conversation_id,
                    'performative': Performatives.confirm
                }
            )
        )

    def reply_to_information_request(self, request_msg: RequestInformationMessage, meta: Dict[str, any]):
        """
        The agent will reply to an information request.
        Either the controller asks for the current_p or the MosaikAgent asks for the p_max of the agent.
        :param request_msg: An instance of a RequestInformationMessage
        :param meta: The dict with all meta information of the message
        """

        # get information from the meta dict
        receiver_addr = meta.get('sender_addr', None)
        receiver_id = meta.get('sender_id', None)
        conversation_id = meta.get('conversation_id', None)

        # check what is requested
        requested_variable = request_msg.requested_information

        if receiver_addr is not None and receiver_id is not None:
            if requested_variable == 'p_max':
                # send a p_max reply
                msg_content = create_msg_content(CurrentPMaxMessage, p_max=self.p_max)
                self.schedule_instant_task(
                    self._container.send_message(
                        content=msg_content,
                        receiver_addr=receiver_addr,
                        receiver_id=receiver_id,
                        create_acl=True,
                        acl_metadata={
                            'sender_id': self.aid,
                            'conversation_id': conversation_id,
                        }
                    )
                )
            elif requested_variable == 'current_p':
                # send a current_p reply to the controller
                msg_content = create_msg_content(CurrentPMessage, current_p=self.current_p)
                self.schedule_instant_task(
                    self._container.send_message(
                        content=msg_content,
                        receiver_addr=receiver_addr, receiver_id=receiver_id,
                        create_acl=True, acl_metadata={
                            'sender_id': self.aid,
                            'conversation_id': conversation_id
                        }
                    )
                )
            else:
                # if something else is requested, we do not want to reply
                self.handle_unknown_message(request_msg, meta)
        else:
            # if there is no sender addr you can not send a reply
            self.handle_unknown_message(request_msg, meta)

    def handle_unknown_message(self, content, meta: Dict[str, Any]):
        """
        loggs unexpected messages
        :param content: The content of the message
        :param meta: The meta information
        """
        logger.warning(f"Agent {self._aid} received an unexpected Message with"
                       f"content: {str(content)} and meta: {meta}.")






class MosaikAgent(Agent):
    """This agent is a gateway between the mosaik API and the WecsAgents.

    It forwards the current state of the simulated WECS to the agents, triggers the controller, waits for the controller
    to be done and then collects new set-points for the simulated WECS from the agents.
    """
    def __init__(self, container: Container):
        super().__init__(container)
        self.agents = {}          # Dict mapping mosaik eid to address (agent_addr, agent_id) (set in mosaik api)
        self.controller = None    # Tuple of addr, aid of Controller (set in mosaik api)
        self._p_max = {}          # Dict mapping from mosaik_aid to Future that is pending until a p_max reply arrives

        # Dict mapping from mosaik_aid to Future. We need this to make sure all agents have received their updates.
        self._updates_received = {}

        # Done if the controller has performed one cycle
        self._controller_done = asyncio.Future()

    async def step(self, inputs):
        """
        This will be called from the mosaik api once per step.

        :param inputs: the input dict from mosaik: {eid_1: {'P': p_value}, eid_2: {'P': p_value}...}
        :return: the output dict: {eid_1: p_max_value, eid_2: p_max_value}
        """

        # 1. reset
        self.reset()

        # 2. update state of wecs agents
        await self.update_agents(inputs)

        # 3. trigger control cycle
        await self.trigger_control_cycle()

        # 4. get p_max from wecs and return it
        return await self.get_p_max()

    def handle_msg(self, content, meta):
        """
        We expect three types of messages:
        1) Confirmation of WecsAgents about State updates
        2) Reply from the Controller after we have asked it to perform a control cycle
        3) Replies from the WecsAgents after we have asked them for their p_max value

        :param content: Conten of the message
        :param meta: Meta information
        """
        content = read_msg_content(content)
        if content is None and meta['performative'] == Performatives.confirm:
            # confirmation about update
            self._updates_received[meta['conversation_id']].set_result(True)
        elif isinstance(content, ControlActionsDone):
            # Reply from the Controller after we have asked it to perform a control cycle
            self._controller_done.set_result(True)
        elif isinstance(content, CurrentPMaxMessage):
            # Replies from the WecsAgents after we have asked them for their p_max value
            self.handle_current_p_max_message(content, meta)

    def handle_current_p_max_message(self, content: CurrentPMaxMessage, meta):
        """

        :param content:
        :param meta:
        """
        mosaik_aid = meta['conversation_id']
        self._p_max[mosaik_aid].set_result(content.p_max)

    async def update_agents(self, data):
        """Update the agents with new data from mosaik."""
        futs = [self.schedule_instant_task(
            self._container.send_message(
                receiver_addr=self.agents[mosaik_eid][0],
                receiver_id=self.agents[mosaik_eid][1],
                content=create_msg_content(UpdateStateMessage, state=input_data),
                acl_metadata={'performative': Performatives.inform, 'conversation_id': mosaik_eid,
                              'sender_id': self.aid},
                create_acl=True,
            )
        )
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        # wait for confirmation
        await asyncio.gather(*[fut for fut in self._updates_received.values()])

    async def trigger_control_cycle(self):
        """

        :return:
        """

        self.schedule_instant_task(self._container.send_message(
            receiver_addr=self.controller[0],
            receiver_id=self.controller[1],
            content=create_msg_content(TriggerControlActions),
            acl_metadata={'sender_id': self.aid, 'sender_addr': self._container.addr},
            create_acl=True
        ))
        await asyncio.wait_for(self._controller_done, timeout=3)

    async def get_p_max(self):
        """Collect new set-points (P_max values) from the agents and return
        them to the mosaik API."""
        futs = [self.schedule_instant_task(
            self._container.send_message(
                receiver_addr=agent_addr,
                receiver_id=aid,
                content=create_msg_content(RequestInformationMessage, requested_information='p_max'),
                acl_metadata={'performative': Performatives.request, 'sender_id': self.aid,
                              'conversation_id': mosaik_aid},
                create_acl=True
            )
        )
            for mosaik_aid, (agent_addr, aid) in self.agents.items()]
        # send messages
        await asyncio.gather(*futs)

        # wait for all replies
        await asyncio.gather(*[fut for fut in self._p_max.values()])

        # return p input
        return {key: value.result() for key, value in self._p_max.items()}

    def reset(self):
        """
        :return:
        """
        self._p_max = {aid: asyncio.Future() for aid in self.agents.keys()}
        self._updates_received = {aid: asyncio.Future() for aid in self.agents.keys()}
        self._controller_done = asyncio.Future()









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
            'params': ['P_rated', 'v_rated', 'v_min', 'v_max'],
            'attrs': ['P'],
        },
    },
}


class MosaikAgents(mosaik_api.Simulator, Agent):
    """
    Interface to mosaik.
    """

    def __init__(self):
        #super().__init__(META)
        #super(mosaik_api.Simulator, self).__init__(META)
        
        mosaik_api.Simulator.__init__(self, META)

        #super().__init__(container)
        # We have a step size of 15 minutes specified in seconds:
        self.step_size = 60 * 15  # seconds
        self.host = 'localhost'
        self.port = 5678

        # This future will be triggered when mosaik calls "stop()":
        self.stopped = asyncio.Future()

        # Set by "run()":
        self.mosaik = None  # Proxy object for mosaik

        self.loop = None
        self.loop = asyncio.get_event_loop()

        # Set in "init()"
        self.sid = None  # Mosaik simulator IDs

        

        # The following will be set in init():
        self.main_container = None  # Root agent main_container
        self.agent_container = None  # Container for agents
        self.controller = None
        self.controller_addr = None
        self.controller_id = None
        self.mosaik_agent = None

        self.loop.run_until_complete(self.create_container())
        Agent.__init__(self, self.main_container)

        # Updated in "setup_done()"
        self.agents = {}  # eid : ((host,port), aid)

        # Set/updated in "setup_done()"
        self.uids = {}  # agent_id: unit_id


        #super().__init__(container)
        self.agents = {}          # Dict mapping mosaik eid to address (agent_addr, agent_id) (set in mosaik api)
        #self.controller = None    # Tuple of addr, aid of Controller (set in mosaik api)
        self._p_max = {}          # Dict mapping from mosaik_aid to Future that is pending until a p_max reply arrives

        # Dict mapping from mosaik_aid to Future. We need this to make sure all agents have received their updates.
        self._updates_received = {}

        # Done if the controller has performed one cycle
        self._controller_done = asyncio.Future()




    def init(self, sid, time_resolution=1., **sim_params):

        

        # we have to get the event loop in order to call async functions from init, create and step
        #self.loop = asyncio.get_event_loop()
        self.sid = sid
        self.loop.run_until_complete(self.create_container_and_agents(**sim_params))
        return META

    async def create_container(self):
        #self.loop = asyncio.get_event_loop()
        self.main_container = await create_container(addr=(self.host, self.port))#Container.factory(addr=(self.host, self.port))


    async def create_container_and_agents(self, controller_config):
        """
        Creates two container, the mosaik agent and the controller
        :param controller_config: Configuration for the controller coming from the init call of mosaik
        """

        # Root main_container for the MosaikAgent and Controller
        #self.main_container = await Container.factory(addr=(self.host, self.port))

        # Start the MosaikAgent and Controller agent in the  main_container
        self.mosaik_agent = self # MosaikAgent(self.main_container)

        #super(Agent, self).__init__(self.main_container)

        self._controller = Controller(self.main_container, **controller_config)
        self.controller_addr = self._controller._container.addr
        self.controller_id = self._controller.aid
        self.mosaik_agent.controller = (self.controller_addr, self.controller_id)


        
        # Create container for the WecsAgents
        #self.agent_container = await Container.factory(addr=(self.host, self.port + 1))

    def create(self, num, model, **model_conf):
        """Create *num* instances of *model* and return a list of entity dicts
        to mosaik."""

        assert model in META['models']
        entities = []
        # Get the number of agents created so far and count from this number
        # when creating new entity IDs:
        n_agents = len(self.mosaik_agent.agents)
        for i in range(n_agents, n_agents + num):
            # Entity data
            eid = 'Agent_%s' % i
            entities.append({'eid': eid, 'type': model})

            # as the event loop is not running here, we have to create the agents via loop.run_unti_complete.
            # That means however, that the agents are not able to receive or send messages unless the loop is
            # triggered from this module.
            #wecs_agent = self.loop.run_until_complete(self.create_mango_agent(self.agent_container, model_conf))
            wecs_agent = self.loop.run_until_complete(self.create_mango_agent(self.main_container, model_conf))
            
            # Store details in agents dict
            #self.mosaik_agent.agents[eid] = (self.agent_container.addr, wecs_agent.aid)
            self.mosaik_agent.agents[eid] = (self.main_container.addr, wecs_agent.aid)

        print(self.mosaik_agent.agents)
        return entities

    async def create_mango_agent(self, container: Container, model_conf) -> MangoAgent:
        """
        Creates a new WECS agent.
        This has to be run within a coroutine, in order to let the agents initially register at the controller
        :param container: The container the agent is running in
        :param model_conf: The model configuration
        :return: An instance of a WecsAgent
        """
        return await MangoAgent.create(
                container=container, controller_addr=self.main_container.addr, controller_id=self.controller_id,#self.controller.aid,
                model_conf=model_conf)

    def setup_done(self):
        """Get the entities that our agents are connected to once the scenario
        setup is done."""
        full_ids = ['%s.%s' % (self.sid, eid) for eid in self.mosaik_agent.agents.keys()]
        relations = yield self.mosaik.get_related_entities(full_ids)
        for full_aid, units in relations.items():
            # We should only be connected to one entity
            assert len(units) == 1
            uid, _ = units.popitem()
            # Create a mapping "agent ID -> unit ID"
            aid = full_aid.split('.')[-1]
            self.uids[aid] = uid

    def finalize(self):
        """

        :return:
        """
        self.loop.run_until_complete(self._shutdown(self.controller, #self.mosaik_agent, 
                                                    self.main_container))#, self.main_container))

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

        # Prepare input data and forward it to the agents:
        data = {}
        for eid, attrs in inputs.items():
            input_data = {}
            for attr, values in attrs.items():
                assert len(values) == 1  # b/c we're only connected to 1 unit
                _, value = values.popitem()
                input_data[attr] = value
                data[eid] = input_data

        # trigger the loop to enable agents to send / receive messages via run_until_complete
        output_dict = self.loop.run_until_complete(self.mosaik_agent._step(inputs=data))

        # Make "set_data()" call back to mosaik to send the set-points:
        inputs = {aid: {self.uids[aid]: {'P_max': P_max}}
                  for aid, P_max in output_dict.items()}
        yield self.mosaik.set_data(inputs)

        return time + self.step_size
    
    def get_data(self, outputs):
        # we are going to send the data asynchronously via set_data, hence we do not need to implement get_data()
        pass



######################################

    async def _step(self, inputs):
        """
        This will be called from the mosaik api once per step.

        :param inputs: the input dict from mosaik: {eid_1: {'P': p_value}, eid_2: {'P': p_value}...}
        :return: the output dict: {eid_1: p_max_value, eid_2: p_max_value}
        """

        # 1. reset
        self.reset()

        # 2. update state of wecs agents
        await self.update_agents(inputs)

        # 3. trigger control cycle
        await self.trigger_control_cycle()

        # 4. get p_max from wecs and return it
        return await self.get_p_max()

    def handle_msg(self, content, meta):
        """
        We expect three types of messages:
        1) Confirmation of WecsAgents about State updates
        2) Reply from the Controller after we have asked it to perform a control cycle
        3) Replies from the WecsAgents after we have asked them for their p_max value

        :param content: Conten of the message
        :param meta: Meta information
        """
        content = read_msg_content(content)
        if content is None and meta['performative'] == Performatives.confirm:
            # confirmation about update
            self._updates_received[meta['conversation_id']].set_result(True)
        elif isinstance(content, ControlActionsDone):
            # Reply from the Controller after we have asked it to perform a control cycle
            self._controller_done.set_result(True)
        elif isinstance(content, CurrentPMaxMessage):
            # Replies from the WecsAgents after we have asked them for their p_max value
            self.handle_current_p_max_message(content, meta)

    def handle_current_p_max_message(self, content: CurrentPMaxMessage, meta):
        """

        :param content:
        :param meta:
        """
        mosaik_aid = meta['conversation_id']
        self._p_max[mosaik_aid].set_result(content.p_max)

    async def update_agents(self, data):
        """Update the agents with new data from mosaik."""

        print(data)
        futs = [self.schedule_instant_task(
            #self._container.send_message(
            self.main_container.send_message(
                receiver_addr=self.agents[mosaik_eid][0],
                receiver_id=self.agents[mosaik_eid][1],
                content=create_msg_content(UpdateStateMessage, state=input_data),
                acl_metadata={'performative': Performatives.inform, 'conversation_id': mosaik_eid,
                              #'sender_id': self.aid},
                              'sender_id': self.agents[mosaik_eid][1]},
                create_acl=True,
            )
        )
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        # wait for confirmation
        await asyncio.gather(*[fut for fut in self._updates_received.values()])

    async def trigger_control_cycle(self):
        """

        :return:
        """
        #self.schedule_instant_task(self._container.send_message(
        self.schedule_instant_task(self.main_container.send_message(
            receiver_addr=self.controller[0],
            receiver_id=self.controller[1],
            content=create_msg_content(TriggerControlActions),
            #acl_metadata={'sender_id': self.aid, 'sender_addr': self._container.addr},
            acl_metadata={'sender_id': self.aid, 
                          'sender_addr': self.main_container.addr},
            create_acl=True
        ))
        await asyncio.wait_for(self._controller_done, timeout=3)

    async def get_p_max(self):
        """Collect new set-points (P_max values) from the agents and return
        them to the mosaik API."""
        futs = [self.schedule_instant_task(
            #self._container.send_message(
            self.main_container.send_message(
                receiver_addr=agent_addr,
                receiver_id=aid,
                content=create_msg_content(RequestInformationMessage, requested_information='p_max'),
                acl_metadata={'performative': Performatives.request, 'sender_id': self.aid,
                              'conversation_id': mosaik_aid},
                create_acl=True
            )
        )
            for mosaik_aid, (agent_addr, aid) in self.agents.items()]
        # send messages
        await asyncio.gather(*futs)

        # wait for all replies
        await asyncio.gather(*[fut for fut in self._p_max.values()])

        # return p input
        return {key: value.result() for key, value in self._p_max.items()}

    def reset(self):
        """
        :return:
        """
        self._p_max = {aid: asyncio.Future() for aid in self.agents.keys()}
        self._updates_received = {aid: asyncio.Future() for aid in self.agents.keys()}
        self._controller_done = asyncio.Future()








if __name__ == '__main__':
    main()
