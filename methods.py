import copy
import pandas as pd
from _mosaik_components.mas.utils import *

MAS_STATE = MAS_DEFAULT_STATE.copy()

def get_cells_data(grid, grid_extra_info, profiles):
    cells = {}
    for e in grid.children:
        if e.eid in grid_extra_info and\
           'name' in grid_extra_info[e.eid] and\
           pd.notna(grid_extra_info[e.eid]['name']):
                name = grid_extra_info[e.eid]['name']
                id = name.split('-')
                if len(id) == 4: # type-index-bus-cell
                    cells.setdefault(id[3], {})
                    cells.setdefault('match_cell', {})
                    cells['match_cell'].update({e.eid : id[3]})
                    cells[id[3]].setdefault(id[0], {})
                    cells[id[3]][id[0]].update({e.eid : {
                        'unit' : e,
                        'type' : id[0],
                        'index' : id[1],
                        'bus' : id[2],
                        'cell' : id[3],
                        'profile' : profiles[name] if name in profiles else {},
                    }})
    return cells

def get_unit_profile(eid, cells_data):
    eid = eid.split('.')[1]
    if eid in cells_data['match_unit']:
        unit_eid = cells_data['match_unit'][eid]
        unit_type = unit_eid.split('-')[0]
        if unit_eid in cells_data['match_cell']:
            return cells_data[cells_data['match_cell'][unit_eid]][unit_type][unit_eid]['profile']

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
        data.setdefault(eid, {})
        for attr in attrs:
            if attr == 'current':
                data[eid].update({'current' : max(output_data[eid]['production'][attr], output_data[eid]['consumption'][attr])})
            elif attr == 'scale_factor':
                if eid in input_data:
                    scale_factor = 1
                    input_value = list(input_data[eid]['current'].values())[0]
                    if input_value != 0:
                        scale_factor = max(output_data[eid]['production']['current'], output_data[eid]['consumption']['current']) / input_value
                    #print(eid, scale_factor)
                    data[eid].update({'scale_factor' : scale_factor})
            else:
                pass
    #print(data)
    return data

    #return {eid: {attr: max(output_data[eid]['production'][attr], output_data[eid]['consumption'][attr]) 
    #                       for attr in attrs
    #                            } for eid, attrs in output_request.items()}

def aggregate_states(requested_states, current_state=MAS_STATE):
    current_state = copy.deepcopy(current_state)
    for aid, state in requested_states.items():
        for i in current_state.keys():
            for j in current_state[i].keys():
                if j == 'scale_factor':
                    current_state[i][j] *= state[i][j]
                else:
                    current_state[i][j] += state[i][j]
    return current_state

def execute_instructions(instruction, current_state, requested_states):
    #print('EXECUTION')
    pass

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
                    grid_state = copy.deepcopy(MAS_STATE)
                if cell_flexibility == None:
                    cell_flexibility = copy.deepcopy(MAS_STATE)

                cell_balance = cell_flexibility['production']['current'] - cell_flexibility['consumption']['current']
                print(highlight('cell balance:', 'red'), cell_balance)
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
                cell_delta = copy.deepcopy(MAS_STATE)
                cell_delta['production']['current'] = cell_inc_production - cell_dec_production
                cell_delta['consumption']['current'] = cell_inc_consumption - cell_dec_consumption     
                grid_delta = copy.deepcopy(MAS_STATE)
                grid_delta['production']['current'] = grid_inc_production - grid_dec_production
                grid_delta['consumption']['current'] = grid_inc_consumption - grid_dec_consumption 

                cell_balance = cell_flexibility['production']['current'] + cell_delta['production']['current'] - cell_flexibility['consumption']['current'] - cell_delta['consumption']['current']
                print(highlight('new cell balance:', 'red'), cell_balance)
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
