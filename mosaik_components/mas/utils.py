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
    'verbose': 1, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': False, # returns wall time of each mosaik step / the core loop execution time 
                                     # as a 'steptime' [sec] output attribute of MosaikAgent 
    'convergence_steps' : 3, # higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence
    'state_dict': MAS_DEFAULT_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': None, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': None, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': None, # method that aggregates gathered states to one top-level state
    'execute_method': None,    # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers,
                                               # executes the received instructions internally
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
