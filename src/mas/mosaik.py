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

from mango.core.agent import Agent
from mango.core.container import Container
from mango.messages.message import Performatives

from src.mas.agent_messages import UpdateStateMessage, RequestInformationMessage, CurrentPMaxMessage, \
    ControlActionsDone, TriggerControlActions, create_msg_content, read_msg_content
from src.mas.controller import Controller
from src.mas.wecs import WecsAgent

logger = logging.getLogger('mas.mosaik')


def main():
    """Run the multi-agent system."""
    logging.basicConfig(level=logging.INFO)
    return mosaik_api.start_simulation(MosaikAPI())


# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'WecsAgent': {
            'public': True,
            'params': ['P_rated', 'v_rated', 'v_min', 'v_max'],
            'attrs': ['P'],
        },
    },
}


class MosaikAPI(mosaik_api.Simulator):
    """
    Interface to mosaik.
    """

    def __init__(self):
        super().__init__(META)
        # We have a step size of 15 minutes specified in seconds:
        self.step_size = 60 * 15  # seconds
        self.host = 'localhost'
        self.port = 5678

        # This future will be triggered when mosaik calls "stop()":
        self.stopped = asyncio.Future()

        # Set by "run()":
        self.mosaik = None  # Proxy object for mosaik

        self.loop = None

        # Set in "init()"
        self.sid = None  # Mosaik simulator IDs

        # The following will be set in init():
        self.main_container = None  # Root agent main_container
        self.agent_container = None  # Container for agents
        self.controller = None
        self.controller_addr = None
        self.controller_id = None
        self.mosaik_agent = None

        # Updated in "setup_done()"
        self.agents = {}  # eid : ((host,port), aid)

        # Set/updated in "setup_done()"
        self.uids = {}  # agent_id: unit_id

    def init(self, sid, time_resolution=1., **sim_params):

        # we have to get the event loop in order to call async functions from init, create and step
        self.loop = asyncio.get_event_loop()
        self.sid = sid
        self.loop.run_until_complete(self.create_container_and_main_agents(**sim_params))
        return META

    async def create_container_and_main_agents(self, controller_config):
        """
        Creates two container, the mosaik agent and the controller
        :param controller_config: Configuration for the controller coming from the init call of mosaik
        """

        # Root main_container for the MosaikAgent and Controller
        self.main_container = await Container.factory(addr=(self.host, self.port))

        # Start the MosaikAgent and Controller agent in the  main_container
        self.mosaik_agent = MosaikAgent(self.main_container)
        self.controller = Controller(self.main_container, **controller_config)
        self.controller_addr = self.controller._container.addr
        self.controller_id = self.controller.aid
        self.mosaik_agent.controller = (self.controller_addr, self.controller_id)

        # Create container for the WecsAgents
        self.agent_container = await Container.factory(addr=(self.host, self.port + 1))

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
            wecs_agent = self.loop.run_until_complete(self.create_wecs_agent(self.agent_container, model_conf))
            # Store details in agents dict
            self.mosaik_agent.agents[eid] = (self.agent_container.addr, wecs_agent.aid)
        return entities

    async def create_wecs_agent(self, container: Container, model_conf) -> WecsAgent:
        """
        Creates a new WECS agent.
        This has to be run within a coroutine, in order to let the agents initially register at the controller
        :param container: The container the agent is running in
        :param model_conf: The model configuration
        :return: An instance of a WecsAgent
        """
        return await WecsAgent.create(
                container=container, controller_addr=self.main_container.addr, controller_id=self.controller.aid,
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
        self.loop.run_until_complete(self._shutdown(self.mosaik_agent, self.controller,
                                                    self.main_container, self.agent_container))

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
        output_dict = self.loop.run_until_complete(self.mosaik_agent.step(inputs=data))

        # Make "set_data()" call back to mosaik to send the set-points:
        inputs = {aid: {self.uids[aid]: {'P_max': P_max}}
                  for aid, P_max in output_dict.items()}
        yield self.mosaik.set_data(inputs)

        return time + self.step_size

    def get_data(self, outputs):
        # we are going to send the data asynchronously via set_data, hence we do not need to implement get_data()
        pass


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


if __name__ == '__main__':
    main()
