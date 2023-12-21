import copy
from pandas.io.json._normalize import nested_to_record    
from termcolor import colored


def state_to_output(output_state):
    return output_state

def input_to_state(input_dict, current_state):
    # state={'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0}}
    _updated_state = copy.deepcopy(current_state)
    for eid, value in input_dict['current'].items():
            if 'Load' in eid: # check consumtion/production
                _updated_state['consumption']['current'] = abs(value)
            elif 'Gen' in eid or 'Wecs' in eid:
                _updated_state['production']['current'] = abs(value)
            elif 'Grid' in eid:
                if value >= 0:
                    _updated_state['production']['current'] = value
                    _updated_state['consumption']['current'] = 0
                else:
                    _updated_state['production']['current'] = 0
                    _updated_state['consumption']['current'] = value
    return _updated_state







def highlight(s):
    return colored(s, 'green')

def reduce_zero_dict(_dict):
    if isinstance(_dict, dict):
        flattened = nested_to_record(_dict, sep='_')
        if sum(flattened.values()) == 0:
            return colored('none', 'blue')
    return _dict 

def reduce_equal_dicts(a_dict, b_dict):
    if isinstance(a_dict, dict) and isinstance(b_dict, dict):
        a_flattened = nested_to_record(a_dict, sep='_')
        b_flattened = nested_to_record(b_dict, sep='_')
        if a_flattened == b_flattened:
            return colored('same', 'blue')
    return a_dict  