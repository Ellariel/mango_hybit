import hashlib
import base64
import numbers
from termcolor import colored
from pandas.io.json._normalize import nested_to_record

PRECISION = 10 ** -6

MAS_DEFAULT_STATE = {
    'production' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
        'scale_factor' : 0,
    },
    'consumption' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
        'scale_factor' : 0,
    },    
}

MAS_DEFAULT_CONFIG = { # see MAS_DEFAULT_CONFIG in utils.py 
    # Required parameters
    'verbose': 1, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': False, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'convergence_steps' : 2, # higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence
    'state_dict': MAS_DEFAULT_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': None, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': None, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'aggregation_method': None, # method that aggregates gathered states to one top-level state
    'execute_method': None,    # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers,
                                               # executes the received instructions internally
    'initialize' : None,
    'finalize' : None,

}

def highlight(s, color='green'):
    return colored(s, color)

def reduce_zero_dict(_dict):
    if isinstance(_dict, dict):
        flattened = nested_to_record(_dict, sep='_')
        if sum(flattened.values()) == 0:
            return colored('none', 'dark_grey')
    return _dict 

def reduce_equal_dicts(a_dict, b_dict):
    if isinstance(a_dict, dict) and isinstance(b_dict, dict):
        a_flattened = nested_to_record(a_dict, sep='_')
        b_flattened = nested_to_record(b_dict, sep='_')
        if a_flattened == b_flattened:
            return colored('same', 'dark_grey')
    return a_dict

def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    
    if isinstance(o, numbers.Number):
        if abs(o) <= PRECISION:
            return 0
        else:
            return f'{o:.6f}'

    return o