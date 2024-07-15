import copy, time, sys
import numpy as np
from mosaik_components.mas.utils import *
from mosaik_components.mas.lib.cohda import COHDA

MAS_STATE = MAS_DEFAULT_STATE.copy()
cohda = None

def initialize(**kwargs):
    global cohda
    cohda = COHDA(muted=True)

def finalize(**kwargs):
    global cohda
    del cohda

def state_to_output(aeid, aid, attrs, current_state, converged, current_time_step, first_time_step, **kwargs):
# inputs {'Agent_3': {'current': {'WecsSim-0.wecs-0': 147.26366926766127}},
# output_state: {'Agent_1': {'production': {'min': 0, 'max': 0, 'current': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0.0}},
# entities: {'Agent_3': 'WecsSim-0.wecs-0', 'Agent_4': 'Grid-0.Load-0',
# Agent_6 ['scale_factor'] {'production': {'min': 0, 'max': 0.02, 'current': 0.02, 'scale_factor': 1.0}, 'consumption': {'min': 0, 'max': 0, 'current': 0, 'scale_factor': 1}}
    data = {}

    #if aeid == 'MosaikAgent':
    #   print('MosaikAgent', current_state, attrs)

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
            if not converged:
                data.update({'scale_factor' : scale_factor})
    return data


def aggregate_states(aeid, aid, requested_states, current_state=MAS_STATE, **kwargs):
    current_state = copy.deepcopy(current_state)
    for aid, state in requested_states.items():
        for i in current_state.keys():
            for j in current_state[i].keys():
                    if isinstance(current_state[i][j], (int, float, list)) and isinstance(state[i][j], (int, float, list)):
                        current_state[i][j] += state[i][j]
    return current_state

def calc_delta(current_state, new_state):
            _new_state = copy.deepcopy(new_state)
            for i in new_state.keys():
                for j in new_state[i].keys():
                    if isinstance(_new_state[i][j], (int, float)):
                        _new_state[i][j] -= current_state[i][j]
            return _new_state

def compose_instructions(agents_info, delta):
        _delta = copy.deepcopy(delta)
        _agents_info = copy.deepcopy(agents_info)
        for aid, state in _agents_info.items():
            for i in _delta.keys():
                #if i != 'info':
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
                    
                    state[i]['scale_factor'] = state[i]['current'] - agents_info[aid][i]['current']#/ if agents_info[aid][i]['current'] > PRECISION else 1

        return _agents_info, _delta

def adjust_instruction(current_state, new_state):
        _current_state = copy.deepcopy(current_state)
        #if abs(new_state['production']['current']) > PRECISION or abs(new_state['production']['scale_factor']) > PRECISION:
        #    key = 'production'
        #elif abs(new_state['consumption']['current']) > PRECISION or abs(new_state['consumption']['scale_factor']) > PRECISION:
        #    key = 'consumption'
        #else:
        #    key = 'production'
        key = 'consumption'
        _current_state[key]['scale_factor'] = current_state[key]['current'] - new_state[key]['current']
        _current_state[key]['current'] = new_state[key]['current']
        key = 'production'
        _current_state[key]['scale_factor'] = current_state[key]['current'] - new_state[key]['current']
        _current_state[key]['current'] = new_state[key]['current']

        return _current_state

def compute_instructions(current_state, **kwargs):

    verbose = kwargs.get('verbose', 0)

    def compute_balance(external_network_state,#=copy.deepcopy(MAS_STATE), 
                                    cells_aggregated_state):#=copy.deepcopy(MAS_STATE)):
                
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
                external_network_default_balance = np.clip(external_network_state['production']['current'] - external_network_state['consumption']['current'],# - cell_balance,
                                                            external_network_min,
                                                            external_network_max)
                if verbose >= 0:
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
                    print(highlight('external network balance:', 'blue'), new_external_network_balance)

                ok = abs(new_cell_balance - cell_expected_balance) < PRECISION
                #ok = abs(cell_balance - cell_expected_balance) < PRECISION
                #ok = abs(cell_balance - new_cell_balance) < PRECISION
                if verbose >= 0:
                    if ok:
                       print(highlight('balanced solution', 'blue'))
                return ok, cells_aggregated_state, cells_aggregated_delta, external_network_state, external_network_delta

    requested_states = kwargs.get('requested_states', {})
    instruction = kwargs.get('instruction', None)

    if instruction != None:
        ok = True
        state = None
        delta = calc_delta(current_state, instruction)
        instructions, remains = compose_instructions(requested_states, delta)
    else:
        cells_aggregated_state = aggregate_states(None, None, requested_states)
        ok, _, cells_delta, state, _ = compute_balance(current_state, cells_aggregated_state)
        instructions, remains = compose_instructions(requested_states, cells_delta)

    return ok, instructions, state

def execute_instructions(aeid, aid, instruction, current_state, requested_states, **kwargs):
    global cohda
    ok = True   

    if not len(requested_states) == 0: # not a leaf agent
        if not aeid == 'MosaikAgent':
            ok, instructions, state = compute_instructions(instruction=instruction, 
                                                                    current_state=current_state,
                                                            requested_states=requested_states, **kwargs)
            #print('instructions', instructions)
            state = MAS_STATE.copy()
        else:
            #print(instruction, current_state, requested_states)
            #print('instruction', instruction)
            #print('current_state', current_state)
            #print('requested_states', requested_states)
            #print('len(requested_states)', len(requested_states))
            #cells_aggregated_state = aggregate_states(None, None, requested_states)
            #ok, _, cells_delta, state, _ = compute_balance(current_state, cells_aggregated_state)
            #instructions, remains = compose_instructions(requested_states, cells_delta)
            #print(instructions)


            
            flexibility = [{'flex_max_power': [current_state['production']['max']],
                            'flex_min_power': [current_state['production']['min']]}]

            total_consumption = 0
            for s in requested_states.values():
                flexibility.append({'flex_max_power': [s['production']['max']],
                                    'flex_min_power': [s['production']['min']]})
                total_consumption += s['consumption']['current']

                print(s)
                print()

            target_schedule = [total_consumption]



            #target_schedule = [0.5, 2.0, 5.0]
            #flex = {'flex_max_power': [3.0, 3.0, 3.0],
            #        'flex_min_power': [0.1, 0.2, 0.3],
            #        }
            print('target_schedule:', target_schedule)
            print('flexibility:', flexibility)
            #sys.exit()
            schedules = cohda.execute(target_schedule=target_schedule,
                                            flexibility=flexibility)
            print('schedules:', schedules)

            state = current_state.copy()
            value = schedules.pop('Agent_0', {})['FlexSchedules'][0]
            state['production']['scale_factor'] = value - state['production']['current']
            state['production']['current'] = value

            #print('state', state)

            #print('requested_states', requested_states)
            instructions = {}
            for (agent_, state_), schedule_ in zip(requested_states.items(), schedules.values()):
                state__ = state_.copy()
                state__['production']['current'] = schedule_['FlexSchedules'][0]
                instructions[agent_] = adjust_instruction(state_, state__)
            #time.sleep(1)

    else:
        instructions = {aid : instruction}
        #print(aid)
        #print(current_state)
        #print(instruction)
        #print()
        state = instruction

    return ok, instructions, state