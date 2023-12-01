import asyncio
import logging
from typing import List, Tuple, Dict
from mango.core.agent import Agent
from mango.messages.message import Performatives
from src.mas.agent_messages import RegisterMessage, TriggerControlActions, ControlActionsDone, \
    RequestInformationMessage, SetPMaxMessage, CurrentPMessage, read_msg_content, create_msg_content

logger = logging.getLogger(__name__)


class Controller(Agent):
    """The Controller agent knows all WecsAgents of its wind farm.

    On request, it collects the current power output of all WECS from the WecsAgents
    and checks if the total power output exceeds *max_windpark_feedin*.
    If so, it sets power limits of the WECS.

    """

    def __init__(self, container, *, max_windpark_feedin):
        """
        Initializes the controller
        :param container: The main_container instance the controller should run in
        :param max_windpark_feedin: The max power of the total windpark
        """
        super().__init__(container)
        self.max_windpark_feedin = max_windpark_feedin

        # List of WecsAgents registered with us
        # It contains a tuple with the agent_addr (as tuple of (host, port)) and the agent_id (as str) for ech agent
        self.wecs: List[Tuple[Tuple[str, int], str]] = []

        # Future indicating whether a control cycle has just been done. Needed to signal mosaik that a step is done.
        self.cycle_done: asyncio.Future = asyncio.Future()
        self.no_cycles: int = 0  # counter of the number of cycles

        # Dicts with conversation_id: Future.
        # The Future is pending as long as we wait for a reply to a
        # RequestInformationMessage for current_p of the wecs.
        self.open_p_requests: Dict[str, asyncio.Future] = {}
        # The Future object is pending as long as we wait for a confirmation of a received
        # SetPMaxMessage
        self.open_confirmations: Dict[str, asyncio.Future] = {}

    def handle_msg(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e. g. performative, sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        if meta.get('performative', None) == Performatives.confirm:
            self.handle_confirmation(content, meta)
        elif isinstance(content, RegisterMessage):
            self.handle_register_msg(content, meta)
        elif isinstance(content, TriggerControlActions):
            # self.perform_control_action is a coroutine, so we need to schedule it as a task
            self.schedule_instant_task(coroutine=self.perform_control_action(content, meta))
        elif isinstance(content, CurrentPMessage):
            self.handle_current_p_msg(content, meta)
        else:
            logger.warning(f"Controller received an unexpected Message with"
                           f"content: {content} and meta: {meta}.")

    def handle_register_msg(self, content: RegisterMessage, meta):
        """
        Registered an agent
        :param content: Register message
        :param meta information
        """
        addr = (content.host, content.port)
        aid = content.aid
        self.wecs.append((addr, aid))
        # reply with a confirmation
        self.schedule_instant_task(self._container.send_message(
            receiver_addr=addr,
            receiver_id=aid,
            content=None,
            acl_metadata={'performative': Performatives.confirm},
            create_acl=True
        ))

    def handle_confirmation(self, content, meta):
        """
        Handles a confirmation message from a wecs that confirms an incoming p_max message
        :param content: Should be None
        :param meta: The meta dict including the conversation_id
        """
        # Set the Result of the corresponding future
        conv_id = meta.get('conversation_id', None)
        if conv_id is not None:
            self.open_confirmations[conv_id].set_result(True)
        else:
            logger.warning(f"Controller received an unexpected Message with"
                           f"content: {content} and meta: {meta}.")

    def handle_current_p_msg(self, content: CurrentPMessage, meta):
        """
        Handles a current p message from a wecs
        :param content: The CurrentPMessage
        :param meta: The meta dict
        """
        current_p = content.current_p
        addr = (meta['sender_addr'], meta['sender_id'])   # addr includes the actual address and the agent_id
        # set the corresponding Future
        self.open_p_requests[meta['conversation_id']].set_result((addr, current_p))

    async def perform_control_action(self, content, meta):
        """
        We will:
        (1) Ask all registered agents for their current p
        (2) Wait for their replies
        (3) Calculate if the cumulated feed-in of all WECS exceeds the given limit for the wind park
        (4) Send new p_max values to all registered agents
        (5) Wait for their confirmations
        (6) Trigger the Future cycle_done to indicate, that one control cycle has been done
        """
        # increase counter
        self.no_cycles += 1
        cycle_no = self.no_cycles

        # Send RequestInformationMessage for current_p to all known agents
        for agent_addr, agent_id in self.wecs:
            # create unique conversation_id
            conversation_id = f'{agent_addr}_{agent_id}_{cycle_no}'
            # create Future to get a signal, when a reply to the p request have arrived
            self.open_p_requests[conversation_id] = asyncio.Future()

            msg_content = create_msg_content(RequestInformationMessage, requested_information='current_p')
            self.schedule_instant_task(
                coroutine=self._container.send_message(
                    content=msg_content,
                    receiver_addr=agent_addr, receiver_id=agent_id, create_acl=True,
                    acl_metadata={
                        'conversation_id': conversation_id,
                        'performative': Performatives.request,
                        'sender_id': self.aid
                    }
                )
            )

        # wait until all replies have arrived
        wecs_feedin = await asyncio.gather(*self.open_p_requests.values())
        current_feedin = sum([i[1] for i in wecs_feedin])

        if current_feedin > self.max_windpark_feedin:
            # Set new power limits *P_max* for all WECS
            factor = self.max_windpark_feedin / current_feedin
            p_max = [(addr, p * factor) for addr, p in wecs_feedin]
            # Check invariant: "sum(i[1] for i in p_max) == max_feedin", but use
            # subtraction to check float-equality
            assert abs(sum(i[1] for i in p_max) - self.max_windpark_feedin) < 0.01

        else:
            p_max = [(addr, None) for addr, _ in wecs_feedin]

        # send new p_max values
        for addr, new_p_max in p_max:
            agent_addr, agent_id = addr
            # create unique conversation_id
            conversation_id = f'{agent_addr}_{agent_id}_{cycle_no}'
            # create Future that is done once we have received a confirmation
            self.open_confirmations[conversation_id] = asyncio.Future()
            self.schedule_instant_task(
                coroutine=self._container.send_message(
                    content=create_msg_content(SetPMaxMessage, p_max=new_p_max),
                    receiver_addr=agent_addr, receiver_id=agent_id, create_acl=True,
                    acl_metadata={
                        'conversation_id': conversation_id,
                        'sender_id': self.aid,
                        'performative': Performatives.inform
                        }
                    ))

        # wait for all confirmations
        await asyncio.gather(*self.open_confirmations.values())

        # clear future dicts now for next cycle
        self.open_confirmations = {}
        self.open_p_requests = {}

        # reply to the trigger if receiver id exists
        if 'sender_id' in meta.keys():
            self.schedule_instant_task(self._container.send_message(
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                content=create_msg_content(ControlActionsDone),
                create_acl=True
            ))
        # signal that step is done - needed for testing
        self.cycle_done.set_result(True)
        # create a new future for the next cycle.
        self.cycle_done = asyncio.Future()
