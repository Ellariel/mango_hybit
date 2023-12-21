import sys, inspect
from dataclasses import dataclass, asdict
from typing import Dict

@dataclass
class GeneralMessage:
    message_id: int
    converted_dict: Dict

@dataclass
class RegisterRequestMessage:
    """
    Agent -> Controller
    """
    host: str
    port: int
    aid: str

@dataclass
class RegisterConfirmMessage:
    """
    Controller -> Agent
    """
    host: str
    port: int
    aid: str

@dataclass
class UpdateStateMessage:
    """
    Mosaik -> Agent
    The state is a dict from mosaik. It should include the keys
    """
    state: Dict

@dataclass
class UpdateConfirmMessage:
    """
    Agent -> Mosaik
    """

@dataclass
class TriggerControlActions:
    """
    Mosaik -> Controller
    """

@dataclass
class ControlActionsDone:
    """
    Controller -> Mosaik
    """
    info: Dict

@dataclass
class RequestStateMessage:
    """
    Controller -> Agents
    """
    state: Dict

@dataclass
class AnswerStateMessage:
    """
    Agents -> Controller
    """
    state: Dict

@dataclass
class BroadcastInstructionsMessage:
    """
    Controller -> Agents
    """
    instructions: Dict

@dataclass
class InstructionsConfirmMessage:
    """
    Agents -> Controller
    """
    instructions: Dict

def get_class(msg_id: int):
    if msg_id >= len(CLSMEMBERS):
        return None
    else:
        return CLSMEMBERS[msg_id]


def get_msg_id(msg_cls):
    if msg_cls not in CLSMEMBERS:
        return None
    else:
        return CLSMEMBERS.index(msg_cls)


def create_msg_content(cls_name, *args, **kwargs):
    # get id
    msg_id = get_msg_id(cls_name)
    if msg_id is None:
        return None
    else:
        return asdict(GeneralMessage(
            message_id=msg_id,
            converted_dict=asdict(cls_name(*args, **kwargs))
        ))


def read_msg_content(msg_dict: dict):
    if not isinstance(msg_dict, dict):
        return msg_dict
    else:
        if len(msg_dict.keys()) != 2 or 'message_id' not in msg_dict.keys() or 'converted_dict' not in msg_dict.keys():
            return msg_dict
        else:
            msg_cls = get_class(msg_dict['message_id'])
            return msg_cls(**msg_dict['converted_dict'])


CLSMEMBERS = [c for _, c in inspect.getmembers(sys.modules[__name__], inspect.isclass)]
