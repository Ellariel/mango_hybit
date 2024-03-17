import copy, sys
import pandas as pd
from _mosaik_components.mas.utils import *

MAS_STATE = MAS_DEFAULT_STATE.copy()
PRECISION = 10 ** -6

def _input_to_state(aeid, aid, input_data, current_state, **kwargs):
    # state={'current': {'Grid-0.Gen-0': 1.0, 'Grid-0.Load-0': 1.0}}
    #global cells
    #input Agent_3 {'current': {'FLSim-0.FLSim-0': 0.9}} {'production': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}, 'consumption': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}}
    
    #cells = {}
    state = copy.deepcopy(MAS_STATE)
    #print(input_data, current_state)
    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            if 'Load' in eid or 'FL' in eid: # check the type of connected unit and its profile
                state['consumption']['current'] += abs(value)
            elif 'Gen' in eid or 'PV' in eid or 'Wecs' in eid:
                state['production']['current'] += abs(value)
            elif 'ExternalGrid' in eid: #{'Grid-0.ExternalGrid-0': 45.30143468767862}} 
                if value > 0: # check the convention here!
                    state['production']['current'] += value
                else:
                    state['consumption']['current'] += abs(value)           
            #print(eid, input_data, state)
            #break

    '''
    if 'current' in input_data:
        for eid, value in input_data['current'].items():
            profile = get_unit_profile(eid, cells)
            if 'Load' in eid or 'FL' in eid: # check the type of connected unit and its profile
                state['consumption']['min'] = profile['min']
                state['consumption']['max'] = profile['max']
                #if value == 0: #input_data.get('scale_factor', 1) == 1 and 
                #    value = random.uniform(profile['min'], profile['max'])
                state['consumption']['current'] = np.clip(abs(value), profile['min'], profile['max'])
                state['production']['current'] = 0
                state['production']['min'] = 0
                state['production']['max'] = 0
            elif 'Gen' in eid or 'PV' in eid or 'Wecs' in eid:
                state['production']['min'] = profile['min']
                state['production']['max'] = min(abs(value), profile['max'])
                state['production']['current'] = np.clip(abs(value), profile['min'], profile['max'])
                state['consumption']['current'] = 0
                state['consumption']['min'] = 0
                state['consumption']['max'] = 0
            elif 'ExternalGrid' in eid:
                if value >= 0: # check the convention here!
                    state['production']['current'] = value
                    state['production']['min'] = value * -3
                    state['production']['max'] = value * 3
                    state['consumption']['current'] = 0
                    state['consumption']['min'] = 0
                    state['consumption']['max'] = 0
                else:
                    state['production']['current'] = 0
                    state['production']['min'] = 0
                    state['production']['max'] = 0
                    state['consumption']['current'] = abs(value)
                    state['consumption']['min'] = abs(value) * -3
                    state['consumption']['max'] = abs(value) * 3
            
            print(eid, input_data, state)
            break
    '''
    print(highlight('\ninput'), aeid, aid, input_data, current_state, state)
    return state

def state_to_output(aeid, aid, attrs, current_state, converged, **kwargs):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
# Agent_6 ['scale_factor'] {'production': {'min': 0, 'max': 0.02, 'current': 0.02, 'scale_factor': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}}
    data = {}
    #print(highlight('\noutputs:'), output_data)
    #print(new_timestep)
    #print(eid, attrs, current_state)

    #print('converged', converged)

    if abs(current_state['production']['current']) > PRECISION or abs(current_state['production']['scale_factor']) > PRECISION:
        key = 'production'
    elif abs(current_state['consumption']['current']) > PRECISION or abs(current_state['consumption']['scale_factor']) > PRECISION:
        key = 'consumption'
    else:
        key = 'production'

    current = current_state[key]['current']
    scale_factor = current_state[key]['scale_factor']

    for attr in attrs:
        if 'current' == attr:
            data.update({'current' : current})
        elif 'scale_factor' == attr:
            if abs(scale_factor) > PRECISION and not converged:
                data.update({'scale_factor' : scale_factor})

    '''
    for eid, attrs in output_request.items():
        data.setdefault(eid, {})
        if eid in output_data:
            scale_factor = 1
            if output_data[eid]['production']['current'] > output_data[eid]['consumption']['current']:
                current = output_data[eid]['production']['current']
                scale_factor = output_data[eid]['production']['scale_factor']
            else:
                current = output_data[eid]['consumption']['current']
                scale_factor = output_data[eid]['consumption']['scale_factor']                
            data[eid].update({'current' : current})
            if abs(scale_factor - 1) > PRECISION:
                data[eid].update({'scale_factor' : scale_factor})
            print(eid, 'scale_factor', scale_factor, 'current', current)
    '''
    #print(data)
    #print(highlight('\noutput'), aeid, aid, attrs, current_state, data)
    return data

'''
def state_to_output(output_request, output_data): #input_data):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
    data = {}
    #print(highlight('\noutputs:'), output_data)
    #print(new_timestep)
    for eid, attrs in output_request.items():
        data.setdefault(eid, {})
        if eid in output_data:
            scale_factor = 1
            if output_data[eid]['production']['current'] > output_data[eid]['consumption']['current']:
                current = output_data[eid]['production']['current']
                scale_factor = output_data[eid]['production']['scale_factor']
            else:
                current = output_data[eid]['consumption']['current']
                scale_factor = output_data[eid]['consumption']['scale_factor']                
            data[eid].update({'current' : current})
            if abs(scale_factor - 1) > PRECISION:
                data[eid].update({'scale_factor' : scale_factor})
            print(eid, 'scale_factor', scale_factor, 'current', current)
    return data
'''

'''
def state_to_output(output_request, output_data, input_data):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
    data = {}
    print(highlight('\noutputs:'), output_data)
    #print(new_timestep)
    for eid, attrs in output_request.items():
        data.setdefault(eid, {})
        if eid in output_data:
            current = max(output_data[eid]['production']['current'], output_data[eid]['consumption']['current'])
            if current < PRECISION:
                current = 0
            for attr in attrs:
                if attr == 'current':
                    data[eid].update({'current' : current})
                elif attr == 'scale_factor':
                    if eid in input_data:
                        input_value = abs(list(input_data[eid]['current'].values())[0])
                        scale_factor = 1 if input_value < PRECISION else current / input_value
                        if abs(scale_factor - 1) > PRECISION:
                            data[eid].update({'scale_factor' : scale_factor})
                        print(eid, 'scale_factor', scale_factor, 'input_value', input_value, 'current', current)
                else:
                    pass
    return data
'''

def aggregate_states(requested_states, current_state=MAS_STATE, **kwargs):
    current_state = copy.deepcopy(current_state)
    for aid, state in requested_states.items():
        for i in current_state.keys():
            for j in current_state[i].keys():
                #if j == 'scale_factor':
                #    current_state[i][j] *= state[i][j]
                #else:
                    if isinstance(current_state[i][j], (int, float, list)):
                        current_state[i][j] += state[i][j]
                    #else:
                    #    print('NOAGGGGG')
    return current_state
'''
def execute_instructions(instruction, current_state, requested_states):
    #print('EXECUTION')
    #print('instruction',instruction)
    #print('current_state',current_state)
    #instruction = copy.deepcopy(instruction)

    new = instruction['production']['current']
    current = current_state['production']['current']
    instruction['production']['scale_factor'] = new / current if current > PRECISION else 1

    new = instruction['consumption']['current']
    current = current_state['consumption']['current']
    instruction['consumption']['scale_factor'] = new / current if current > PRECISION else 1

    #print('instruction', instruction)
    print('EXECUTION scale_factr', instruction['production']['scale_factor'], instruction['consumption']['scale_factor'], requested_states.keys())

    return instruction
'''

def compute_instructions(current_state, **kwargs):
    verbose = kwargs.get('verbose', 0)
    def calc_delta(current_state, new_state):
            _new_state = copy.deepcopy(new_state)
            for i in new_state.keys():
                for j in new_state[i].keys():
                    if isinstance(_new_state[i][j], (int, float)):
                        _new_state[i][j] -= current_state[i][j]
                    #else:
                    #    print('NONUM deta')
            return _new_state
    #def add_delta(current_state, delta):
    #        new_state = copy.deepcopy(current_state)
    #        for i in delta.keys():
    #            for j in delta[i].keys():
    #                new_state[i][j] += delta[i][j]
    #        return new_state
    def compose_instructions(agents_info, delta):
        _delta = copy.deepcopy(delta)
        _agents_info = copy.deepcopy(agents_info)
        for aid, state in _agents_info.items():
            for i in _delta.keys():
                if i != 'info':
                    max_inc = state[i]['max'] - state[i]['current']
                    max_dec = state[i]['current'] - state[i]['min']
                    if _delta[i]['current'] > PRECISION:
                        if _delta[i]['current'] <= max_inc:
                            state[i]['current'] += _delta[i]['current']
                            _delta[i]['current'] = 0
                        else:
                            state[i]['current'] += max_inc
                            _delta[i]['current'] -= max_inc
                    elif _delta[i]['current'] <= PRECISION:            
                        if abs(_delta[i]['current']) <= max_dec:
                            state[i]['current'] += _delta[i]['current']
                            _delta[i]['current'] = 0
                        else:
                            state[i]['current'] -= max_dec
                            _delta[i]['current'] += max_dec
                    #state[i]['scale_factor'] = state[i]['current'] / agents_info[aid][i]['current'] if agents_info[aid][i]['current'] > PRECISION else 1
                    
                    #print(aid, 'scale_factor old', agents_info[aid][i]['scale_factor'], ' current', agents_info[aid][i]['current'])
                    
                    #state[i]['scale_factor'] = state[i]['current'] - (agents_info[aid][i]['current'] - agents_info[aid][i]['scale_factor'])#/ if agents_info[aid][i]['current'] > PRECISION else 1
                    state[i]['scale_factor'] = state[i]['current'] - agents_info[aid][i]['current']#/ if agents_info[aid][i]['current'] > PRECISION else 1


                    #print(aid, 'scale_factor new', state[i]['scale_factor'], ' current', state[i]['current'])
        
        return _agents_info, _delta
    
    def compute_balance(external_network_state=copy.deepcopy(MAS_STATE), 
                                    cells_aggregated_state=copy.deepcopy(MAS_STATE)):
                
                #cells_aggregated_state = copy.deepcopy(cells_aggregated_state)
                #external_network_state = copy.deepcopy(external_network_state)               
                old_cells_aggregated_state = copy.deepcopy(cells_aggregated_state)
                old_external_network_state = copy.deepcopy(external_network_state)
                cell_balance = cells_aggregated_state['production']['current'] - cells_aggregated_state['consumption']['current']
                cells_aggregated_state['production']['current'] = np.clip(cells_aggregated_state['production']['current'], 
                                                                            cells_aggregated_state['production']['min'], 
                                                                            cells_aggregated_state['production']['max'])
                cells_aggregated_state['consumption']['current'] = np.clip(cells_aggregated_state['consumption']['current'], 
                                                                            cells_aggregated_state['consumption']['min'], 
                                                                            cells_aggregated_state['consumption']['max'])
                cell_expected_balance = cells_aggregated_state['production']['current'] - cells_aggregated_state['consumption']['current']

                external_network_min = external_network_state['consumption']['max'] * (-1)
                external_network_max = external_network_state['production']['max']
                external_network_default_balance = np.clip(external_network_state['production']['current'] - external_network_state['consumption']['current'] - cell_balance,
                                                            external_network_min,
                                                            external_network_max)
                if verbose >= 0:
                    #print(highlight('cell balance:', 'red'), cell_balance)
                    print(highlight('cells balance:', 'red'), cell_balance)
                    print(highlight('cells expected balance:', 'red'), cell_expected_balance)
                    print(highlight('external network default balance:', 'blue'), external_network_default_balance)
                #print('flexibility:')
                #print('external_network_min', external_network_min)
                #print('external_network_max', external_network_max)

                cell_inc_production = cells_aggregated_state['production']['max'] - cells_aggregated_state['production']['current']
                if cell_inc_production < PRECISION:
                    cell_inc_production = 0
                cell_dec_production = cells_aggregated_state['production']['current'] - cells_aggregated_state['production']['min']
                if cell_dec_production < PRECISION:
                    cell_dec_production = 0
                cell_dec_consumption = cells_aggregated_state['consumption']['current'] - cells_aggregated_state['consumption']['min']
                if cell_dec_consumption < PRECISION:
                    cell_dec_consumption = 0
                cell_inc_consumption = cells_aggregated_state['consumption']['max'] - cells_aggregated_state['consumption']['current']
                if cell_inc_consumption < PRECISION:
                    cell_inc_consumption = 0
                
                #print('cell_inc_production', cell_inc_production)
                #print('cell_dec_production', cell_dec_production)
                #print('cell_inc_consumption', cell_inc_consumption)
                #print('cell_dec_consumption', cell_dec_consumption)
                #print()

                grid_inc = external_network_max - external_network_default_balance
                grid_dec = external_network_default_balance - external_network_min
                #print('grid_inc', grid_inc)
                #print('grid_dec', grid_dec)

                if cell_expected_balance < PRECISION:
                    if abs(cell_expected_balance) <= cell_inc_production:
                        cell_inc_production = abs(cell_expected_balance)
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0
                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    elif abs(cell_expected_balance) <= cell_inc_production + cell_dec_consumption:
                        cell_dec_consumption = abs(cell_expected_balance) - cell_inc_production
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0
                        
                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    #elif abs(cell_expected_balance) <= cell_inc_production + cell_dec_consumption + grid_inc_production:
                    #    grid_inc_production = abs(cell_expected_balance) - cell_inc_production - cell_dec_consumption
                    elif abs(cell_expected_balance) <= cell_inc_production + cell_dec_consumption + grid_inc:
                        grid_inc = abs(cell_expected_balance) - cell_inc_production - cell_dec_consumption
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0
                        #grid_inc = 0
                        grid_dec = 0
                        
                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    #elif abs(cell_expected_balance) <= cell_inc_production + cell_dec_consumption + grid_inc_production + grid_dec_consumption:
                    #    grid_dec_consumption = abs(cell_expected_balance) - cell_inc_production - cell_dec_consumption - grid_inc_production
                    elif abs(cell_expected_balance) <= cell_inc_production + cell_dec_consumption + grid_inc + grid_dec:
                        grid_dec = abs(cell_expected_balance) - cell_inc_production - cell_dec_consumption - grid_inc
                        #cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        #cell_dec_consumption = 0
                        #grid_inc = 0
                        #grid_dec = 0
                        
                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    else:
                        if verbose >= 0:
                            print('balance mismatch!')
                        cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0

                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0 
                elif cell_expected_balance > PRECISION:
                    if abs(cell_expected_balance) <= cell_dec_production:
                        cell_dec_production = abs(cell_expected_balance)
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0

                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    elif abs(cell_expected_balance) <= cell_dec_production + cell_inc_consumption:
                        cell_inc_consumption = abs(cell_expected_balance) - cell_dec_production
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        #cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0

                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    #elif abs(cell_expected_balance) <= cell_dec_production + cell_inc_consumption + grid_dec_production:
                    #    grid_dec_production = abs(cell_expected_balance) - cell_dec_production - cell_inc_consumption
                    elif abs(cell_expected_balance) <= cell_dec_production + cell_inc_consumption + grid_dec:
                        grid_dec = abs(cell_expected_balance) - cell_dec_production - cell_inc_consumption
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        #cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        #grid_dec = 0

                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0
                    #elif abs(cell_expected_balance) <= cell_dec_production + cell_inc_consumption + grid_dec_production + grid_inc_consumption:
                    #    grid_inc_consumption = abs(cell_expected_balance) - cell_dec_production - cell_inc_consumption - grid_dec_production
                    elif abs(cell_expected_balance) <= cell_dec_production + cell_inc_consumption + grid_dec + grid_inc:
                        grid_inc = abs(cell_expected_balance) - cell_dec_production - cell_inc_consumption - grid_dec
                        cell_inc_production = 0
                        #cell_dec_production = 0
                        #cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        #grid_inc = 0
                        #grid_dec = 0
                    else:
                        if verbose >= 0:
                            print('balance mismatch!')
                        cell_inc_production = 0
                        cell_dec_production = 0
                        cell_inc_consumption = 0
                        cell_dec_consumption = 0
                        grid_inc = 0
                        grid_dec = 0
                        #grid_inc_production = 0
                        #grid_dec_production = 0
                        #grid_inc_consumption = 0
                        #grid_dec_consumption = 0 
                else:
                    cell_inc_production = 0
                    cell_dec_production = 0
                    cell_inc_consumption = 0
                    cell_dec_consumption = 0
                    grid_inc = 0
                    grid_dec = 0
                    if verbose >= 0:
                        print('perfect balance!')  
                #print()           
                #print('control action:')
                #print('cell_inc_production', cell_inc_production) 
                #print('cell_dec_production', cell_dec_production) 
                #print('cell_dec_consumption', cell_dec_consumption) 
                #print('cell_inc_consumption', cell_inc_consumption) 
                #print('grid_inc', grid_inc) 
                #print('grid_dec', grid_dec) 

                #print('grid_inc_production', grid_inc_production)
                #print('grid_dec_production', grid_dec_production)
                #print('grid_dec_consumption', grid_dec_consumption) 
                #print('grid_inc_consumption', grid_inc_consumption)

                cells_aggregated_state['production']['current'] = cells_aggregated_state['production']['current'] + cell_inc_production - cell_dec_production
                cells_aggregated_state['consumption']['current'] = cells_aggregated_state['consumption']['current'] + cell_inc_consumption - cell_dec_consumption     
                new_cell_balance = cells_aggregated_state['production']['current'] - cells_aggregated_state['consumption']['current']
                
                #external_network_state['production']['current'] = external_network_state['production']['current'] + grid_inc - grid_dec
                #external_network_state['consumption']['current'] = external_network_state['consumption']['current'] + grid_inc - grid_dec
                new_external_network_balance = external_network_default_balance + grid_inc - grid_dec
                if new_external_network_balance > PRECISION:
                    external_network_state['production']['current'] = new_external_network_balance
                    external_network_state['consumption']['current'] = 0
                else:
                    external_network_state['production']['current'] = 0
                    external_network_state['consumption']['current'] = abs(new_external_network_balance)      

                cells_aggregated_delta = calc_delta(old_cells_aggregated_state, cells_aggregated_state)
                external_network_delta = calc_delta(old_external_network_state, external_network_state)

                #print('new_cells_aggregated_state', cells_aggregated_state)
                #print('new_external_network_state', external_network_state)
                #print('cells_aggregated_delta', cells_aggregated_delta)
                #print('external_network_delta', external_network_delta)

                if verbose >= 0:
                    print(highlight('new cells balance:', 'red'), new_cell_balance)
                    print(highlight('new external network balance:', 'blue'), new_external_network_balance)
                #print('grid delta:', grid_delta)
                #print('grid delta:', grid_delta)

                ok = abs(cell_balance - cell_expected_balance) < PRECISION
                if verbose >= 0:
                    if ok:
                        print(highlight('CONVERGED!', 'green'))
                    else:
                        print(highlight('NOT CONVERGED!', 'red'))
                return ok, cells_aggregated_state, cells_aggregated_delta, external_network_state, external_network_delta

    requested_states = kwargs.get('requested_states', {})
    instruction = kwargs.get('instruction', None)

    if instruction != None:
        ok = True
        delta = calc_delta(current_state, instruction)
        instructions, remains = compose_instructions(requested_states, delta)
    else:
        cells_aggregated_state = aggregate_states(requested_states)
        #if True:# verbose:
            #print('\ngrid_state:', current_state)
            #print('aggregated_cell_state:', aggregated_state)
        #return cells_aggregated_state, cells_aggregated_delta, external_network_state, external_network_delta
        ok, _, cells_delta, delta, _ = compute_balance(current_state, cells_aggregated_state)
        #print('compute_delta_state')
        #print('grid_delta', grid_delta)
        #print('delta', delta)
        instructions, remains = compose_instructions(requested_states, cells_delta)
        #print('instructions', instructions)
        #print('info', reduce_zero_dict(delta))
        #sys.exit()

    return ok, instructions, delta
