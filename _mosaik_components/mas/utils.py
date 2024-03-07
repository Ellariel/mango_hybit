import os
import random
import numpy as np
from termcolor import colored
from pandas.io.json._normalize import nested_to_record

MAS_DEFAULT_STATE = {
    'production' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
        'scale_factor' : 1,
    },
    'consumption' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
        'scale_factor' : 1,
    },    
}

MAS_DEFAULT_CONFIG = {
    'verbose': 1, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'performance': True, # returns wall time of each mosaik step / the core loop execution time 
                         # as a 'steptime' [sec] output attribute of MosaikAgent 
    'state_dict': MAS_DEFAULT_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': None, # method that transforms mosaik inputs dict to the agent state (see `update_state`, default: copy dict)
    'output_method': None, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': None, # method that aggregates gathered states to one top-level state
    'redispatch_method': None, # method that computes and decomposes the redispatch instructions 
                               # that will be hierarchically transmitted from each agent to its connected peers
    'execute_method': None,    # executes the received instructions internally
}

def set_random_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_random_seed(base_seed=None, fixed_range=1000):
    if base_seed:
        set_random_seed(base_seed)
    return random.randint(0, fixed_range)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

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

def hard_close_file_descriptors():
    KEEP_FD = set([0, 1, 2])
    if os.name == 'posix':
        try:
            descs = os.listdir(os.path.join("/proc", str(os.getpid()), "fd"))
            print(f"file descriptors: {len(descs)}")
            for fd in descs:
                if int(fd) not in KEEP_FD:
                    try:
                        print(os.readlink(f"/proc/self/fd/{fd}"))
                        os.close(int(fd))
                    except OSError:
                        pass
        except:
            pass