import copy
from pandas.io.json._normalize import nested_to_record    
from termcolor import colored

MAS_DEFAULT_STATE = {
    'production' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
    },
    'consumption' : {
        'min' : 0,
        'max' : 0,
        'current' : 0,
    },    
}

def state_to_output(output_state):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
    #print(entities)
    data = {}
    for k, v in output_state.items():
        current = output_state[k]['production']['current'] - output_state[k]['consumption']['current']
        data[k] = {'current' : current}
    #print()
   # print(data)
    return data

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

def aggregate_states(requested_states, current_state=MAS_DEFAULT_STATE):
    current_state = copy.deepcopy(current_state)
    for aid, state in requested_states.items():
        for i in current_state.keys():
            for j in current_state[i].keys():
                current_state[i][j] += state[i][j]
    return current_state

def compute_instructions(current_state, **kwargs):
    def calc_delta(current_state, new_state):
            _new_state = copy.deepcopy(new_state)
            for i in new_state.keys():
                for j in new_state[i].keys():
                    _new_state[i][j] -= current_state[i][j]
            return _new_state
    def add_delta(current_state, delta):
            new_state = copy.deepcopy(current_state)
            for i in delta.keys():
                for j in delta[i].keys():
                    new_state[i][j] += delta[i][j]
            return new_state
    def compose_instructions(agents_info, delta):
        _delta = copy.deepcopy(delta)
        _agents_info = copy.deepcopy(agents_info)
        for aid, state in _agents_info.items():
            for i in _delta.keys():
                max_inc = state[i]['max'] - state[i]['current']
                max_dec = state[i]['current'] - state[i]['min']
                if _delta[i]['current'] > 0:
                    if _delta[i]['current'] <= max_inc:
                       state[i]['current'] += _delta[i]['current']
                       _delta[i]['current'] = 0
                    else:
                       state[i]['current'] += max_inc
                       _delta[i]['current'] -= max_inc
                elif _delta[i]['current'] < 0:            
                    if abs(_delta[i]['current']) <= max_dec:
                       state[i]['current'] += _delta[i]['current']
                       _delta[i]['current'] = 0
                    else:
                       state[i]['current'] -= max_dec
                       _delta[i]['current'] += max_dec
        return _agents_info, _delta
    def compute_delta_state(grid_state=None, cell_flexibility=None):
                if grid_state == None:
                    grid_state = copy.deepcopy(MAS_DEFAULT_STATE)
                if cell_flexibility == None:
                    cell_flexibility = copy.deepcopy(MAS_DEFAULT_STATE)

                cell_balance = cell_flexibility['production']['current'] - cell_flexibility['consumption']['current']
                print('cell_balance', cell_balance)
                cell_inc_production = cell_flexibility['production']['max'] - cell_flexibility['production']['current']
                if cell_inc_production < 0:
                    cell_inc_production = 0
                cell_dec_production = cell_flexibility['production']['current'] - cell_flexibility['production']['min']
                if cell_dec_production < 0:
                    cell_dec_production = 0

                cell_dec_consumption = cell_flexibility['consumption']['current'] - cell_flexibility['consumption']['min']
                if cell_dec_consumption < 0:
                    cell_dec_consumption = 0
                cell_inc_consumption = cell_flexibility['consumption']['max'] - cell_flexibility['consumption']['current']
                if cell_inc_consumption < 0:
                    cell_inc_consumption = 0

                grid_inc_production = grid_state['production']['max'] - grid_state['production']['current']
                if grid_inc_production < 0:
                    grid_inc_production = 0
                grid_dec_production = grid_state['production']['current'] - grid_state['production']['min']
                if grid_dec_production < 0:
                    grid_dec_production = 0

                grid_dec_consumption = grid_state['consumption']['current'] - grid_state['consumption']['min']
                if grid_dec_consumption < 0:
                    grid_dec_consumption = 0
                grid_inc_consumption = grid_state['consumption']['max'] - grid_state['consumption']['current']
                if grid_inc_consumption < 0:
                    grid_inc_consumption = 0

                if cell_balance < 0:
                    if abs(cell_balance) <= cell_inc_production:
                        cell_inc_production = abs(cell_balance)
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_inc_production + cell_dec_consumption:
                        cell_dec_consumption = abs(cell_balance) - cell_inc_production
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_inc_production + cell_dec_consumption + grid_inc_production:
                        grid_inc_production = abs(cell_balance) - cell_inc_production - cell_dec_consumption
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0

                        #grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_inc_production + cell_dec_consumption + grid_inc_production + grid_dec_consumption:
                        grid_dec_consumption = abs(cell_balance) - cell_inc_production - cell_dec_consumption - grid_inc_production
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0

                        #grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    else:
                        pass
                        print('balance mismatch!')
                        cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0 
                elif cell_balance > 0:
                    if abs(cell_balance) <= cell_dec_production:
                        cell_dec_production = abs(cell_balance)
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_dec_production + cell_inc_consumption:
                        cell_inc_consumption = abs(cell_balance) - cell_dec_production
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        
                        #cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_dec_production + cell_inc_consumption + grid_dec_production:
                        grid_dec_production = abs(cell_balance) - cell_dec_production - cell_inc_consumption
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        
                        #cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        #grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0
                    elif abs(cell_balance) <= cell_dec_production + cell_inc_consumption + grid_dec_production + grid_inc_consumption:
                        grid_inc_consumption = abs(cell_balance) - cell_dec_production - cell_inc_consumption - grid_dec_production
                    else:
                        pass
                        print('balance mismatch!')
                        cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0 
                else:
                        cell_inc_production = 0
                        cell_dec_production = 0
                        
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0

                        grid_inc_production = 0
                        grid_dec_production = 0
                        
                        grid_inc_consumption = 0
                        grid_dec_consumption = 0  
                        print('perfect balance!')             
                        
                
                #print('cell_inc_production', cell_inc_production) 
                #print('cell_dec_production', cell_dec_production) 
                #print('cell_dec_consumption', cell_dec_consumption) 
                #print('cell_inc_consumption', cell_inc_consumption) 
                #print('grid_inc_production', grid_inc_production)
                #print('grid_dec_production', grid_dec_production)
                #print('grid_dec_consumption', grid_dec_consumption) 
                #print('grid_inc_consumption', grid_inc_consumption) 
                cell_delta = copy.deepcopy(MAS_DEFAULT_STATE)
                cell_delta['production']['current'] = cell_inc_production - cell_dec_production
                cell_delta['consumption']['current'] = cell_inc_consumption - cell_dec_consumption     
                grid_delta = copy.deepcopy(MAS_DEFAULT_STATE)
                grid_delta['production']['current'] = grid_inc_production - grid_dec_production
                grid_delta['consumption']['current'] = grid_inc_consumption - grid_dec_consumption 
                return grid_delta, cell_delta 

    requested_states = kwargs.get('requested_states', None)
    instruction = kwargs.get('instruction', None)

    if instruction != None:
        delta = calc_delta(current_state, instruction)
        instructions, delta_remained = compose_instructions(requested_states, delta)
    else:
        aggregated_state = aggregate_states(requested_states)
        print('grid_state:', current_state)
        print('aggregated_cell_state:', aggregated_state)
        grid_delta, delta = compute_delta_state(current_state, aggregated_state)
        instructions, delta_remained = compose_instructions(requested_states, delta)

    return instructions, delta

MAS_DEFAULT_CONFIG = {
    'verbose': 1,
    'state_dict': MAS_DEFAULT_STATE,
    'input_method': input_to_state,
    'output_method': state_to_output,
    'states_agg_method': aggregate_states,
    'redispatch_method': compute_instructions,
}

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