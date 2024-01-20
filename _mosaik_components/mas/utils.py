import copy
from pandas.io.json._normalize import nested_to_record    
from termcolor import colored

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

def state_to_output(output_request, output_data, input_data):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
    #print(entities)
    #data = {}
    #print(input_data)
    #print(output_data)
    #print(output_request)

    data = {}
    for eid, attrs in output_request.items():
        data[eid] = {}
        for attr in attrs:
            if attr == 'current':
                data[eid].update({'current' : max(output_data[eid]['production'][attr], output_data[eid]['consumption'][attr])})
            elif attr == 'scale_factor':
                if eid in input_data:
                    scale_factor = max(output_data[eid]['production']['current'], output_data[eid]['consumption']['current']) / list(input_data[eid]['current'].values())[0]
                    print(eid, scale_factor)
                    data[eid].update({'scale_factor' : scale_factor})
            else:
                pass
    #print(data)
    return data

    #return {eid: {attr: max(output_data[eid]['production'][attr], output_data[eid]['consumption'][attr]) 
    #                       for attr in attrs
    #                            } for eid, attrs in output_request.items()}

def input_to_state(input_data, current_state):
    # state={'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0}}
    _updated_state = copy.deepcopy(current_state)
    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            #print(eid)
            if 'Load' in eid or 'load' in eid: # check consumtion/production
                _updated_state['consumption']['current'] = abs(value)
                _updated_state['consumption']['min'] = abs(value) * 0.5
                _updated_state['consumption']['max'] = abs(value) * 3
            elif 'Gen' in eid or 'Wecs' in eid or 'PV' in eid:
                _updated_state['production']['current'] = abs(value)
                _updated_state['production']['max'] = abs(value) * 3
                _updated_state['production']['min'] = 0

            elif 'Grid-0.0-Bus 0' in eid:
                if value >= 0:
                    _updated_state['production']['current'] = value
                    _updated_state['production']['max'] = value * 3
                    _updated_state['consumption']['current'] = 0
                    _updated_state['consumption']['max'] = 0
                    
                else:
                    _updated_state['production']['current'] = 0
                    _updated_state['production']['max'] = 0
                    _updated_state['consumption']['current'] = value
                    _updated_state['consumption']['max'] = value * 3
    return _updated_state

def aggregate_states(requested_states, current_state=MAS_DEFAULT_STATE):
    current_state = copy.deepcopy(current_state)
    for aid, state in requested_states.items():
        for i in current_state.keys():
            for j in current_state[i].keys():
                if j == 'scale_factor':
                    current_state[i][j] *= state[i][j]
                else:
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
                print(highlight('cell balance:'), cell_balance)
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

                cell_balance = cell_flexibility['production']['current'] + cell_delta['production']['current'] - cell_flexibility['consumption']['current'] - cell_delta['consumption']['current']
                print(highlight('new cell balance:'), cell_balance)
                print('grid delta:', grid_delta)

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
    'verbose': 1, # 0 - no messages, 1 - basic agent comminication, 2 - full
    'state_dict': MAS_DEFAULT_STATE, # how an agent state that are gathered and comunicated should look like
    'input_method': input_to_state, # method that transforms mosaik inputs dict to the agent state (default: copy dict)
    'output_method': state_to_output, # method that transforms the agent state to mosaik outputs dict (default: copy dict)
    'states_agg_method': aggregate_states, # method that aggregates gathered states to one top-level state
    'redispatch_method': compute_instructions, # method that computes and decomposes the redispatch instructions 
                                               # that will be hierarchically transmitted from each agent to its connected peers
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