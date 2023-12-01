import asyncio
import logging
from typing import Tuple, Dict, Any
from mango.core.agent import Agent
from mango.core.container import Container
from mango.messages.message import Performatives
from src.mas.agent_messages import UpdateStateMessage, SetPMaxMessage, RequestInformationMessage, RegisterMessage, \
    CurrentPMessage, CurrentPMaxMessage, create_msg_content, read_msg_content

logger = logging.getLogger(__name__)

# To make the difference between WECS simulation model and WECS agent more
# clear, this class has the suffix "Agent".  In a real project, I would just
# name the class "WECS"


class WecsAgent(Agent):
    """
    A WecsAgent is the “brain” of a simulated or real WECS.
    """

    def __init__(self, container: Container, controller_addr, controller_id, model_conf: Dict):
        super().__init__(container)

        self.controller = (controller_addr, controller_id)
        self.model_conf = model_conf
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

        wecs = cls(container=container, controller_addr=controller_addr, controller_id=controller_id,
                   model_conf=model_conf)
        await wecs._register(controller_addr=controller_addr, controller_id=controller_id)

        return wecs

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
        This should be called once a wecs agent is initialized

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
