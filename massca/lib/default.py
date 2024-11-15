import sys
import copy
import numbers
import numpy as np
from ..mosaik_agents import META
from ..utils import ZERO, highlight
from .cohda import COHDA
from .swarm import SWARM


DEFAULT_STATE = {
    'production' : {
        'min' : 0,  # flexibility representation (min/max)
        'max' : 0,
        'level' : 0, # current power flow measure (level)
        'delta' : 0, # last/current control action (delta of change relative to level)
    },
    'consumption' : {
        'min' : 0,
        'max' : 0,
        'level' : 0,
        'delta' : 0,
    },    
}

DEFAULT_META = copy.deepcopy(META) # defined names for attributes, which will be used in input/output_method()
DEFAULT_META['models']['MosaikAgents']['attrs'] += ['production_delta[MW]', 
                                                    'consumption_delta[MW]', 
                                                    'consumption[MW]', 
                                                    'production[MW]']

DEFAULT_CONFIG = {
    'META': copy.deepcopy(DEFAULT_META), # returned by init()
    'state_dict': copy.deepcopy(DEFAULT_STATE), # how an agent state, that are gathered and communicated, should look like
    
    'initialize_method' : None, # method that is called before MAS is started, in setup_done()
    'finalize_method' : None, # method that is called after MAS is destroyed, in finalize()

    'input_method': None, # method that transforms mosaik inputs to an agent state (see `update_state`, default behavior: copy dict)
    'output_method': None, # method that transforms an agent state to mosaik outputs (default behavior: copy dict)
    'aggregation_method': None, # method that aggregates states to one state at each hierachical level 
    'execute_method': None, # method that computes and decomposes the control instructions 
                                # that will be transmitted from each agent to its connected peers or underlying agents,
                                # for leaf agents it executes the received instructions internally
    # communication protocols/algorithms passed to execute_method() in the default implementation:
    'between_cells' : 'default', # executed by MosaikAgent
    'within_cell' : 'default', # executed by hierachical agent

    'verbose': 1,   # 0 - no messages, 1 - basic agent comminication messages, 2 - full report
    'performance': False,   # returns wall time of each mosaik step (the core loop execution time) 
                            # as a 'steptime' [sec] output attribute 
    'convergence_steps' : 1, # minimum number of succesfully converged colls of 'execute_method'
                             # to finish the step, higher value ensures convergence
    'convegence_max_steps' : 5, # raise an error if there is no convergence after x convergence_steps
}


def initialize_protocol(proto, **kwargs):
    if callable(proto):
            proto = proto(**kwargs)
    elif isinstance(proto, str):
        if proto == 'cohda':
            proto = COHDA(**kwargs)
        elif proto == 'swarm':
            proto = SWARM(**kwargs)
        elif proto == 'default':
            pass
        else:
            raise NotImplementedError(f"Protocol '{proto}' was not recognized.")
    else:
        raise NotImplementedError(f"Protocol '{proto}' can't be initialized.")
    return proto


class Default():
    def __init__(self, **kwargs):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config['initialize_method'] = self.initialize_method
        self.config['finalize_method'] = self.finalize_method
        self.config['input_method'] = self.input_method
        self.config['output_method'] = self.output_method
        self.config['aggregation_method'] = self.aggregation_method 
        self.config['execute_method'] = self.execute_method
        self.config.update(kwargs) 
        self.cache = {}

    def build(self, **kwargs):
        self.config.update(kwargs)
        self.config['between_cells'] = initialize_protocol(self.config.get('between_cells', 'default'),
                                                           **self.config)
        self.config['within_cell'] = initialize_protocol(self.config.get('within_cell', 'default'),
                                                         **self.config)     
        return self.config
        
    def __del__(self):
        del self.cache
        del self.config['between_cells']
        del self.config['within_cell']

    def input_method(self, aeid, aid, input_data, current_state, current_time, first_time_step, **kwargs):
        return copy.deepcopy(input_data)

    def initialize_method(self, **kwargs):
        pass

    def finalize_method(self, **kwargs):
        pass

    def output_method(self, aeid, aid, attrs, current_state, converged, current_time_step, first_time_step, **kwargs):
        data = {}
        for attr in attrs:
            if attr == 'production[MW]':
                data.update({attr : current_state['production']['level']})
            elif attr == 'production_delta[MW]' and first_time_step and not converged:
                data.update({attr : current_state['production']['delta']})
            elif attr == 'consumption[MW]':
                data.update({attr : current_state['consumption']['level']})
            elif attr == 'consumption_delta[MW]' and first_time_step and not converged:
                data.update({attr : current_state['consumption']['delta']})
        return data

    def update_flexibility(self, current_state, new_state):
            current_state['max'] = new_state['max']
            current_state['min'] = new_state['min']
            return current_state

    def aggregation_method(self, aeid, aid, requested_states, current_state=None, **kwargs):
        if current_state==None:
            current_state = copy.deepcopy(self.config['state_dict'])
        for aid, state in requested_states.items():
            for i in current_state.keys():
                for j in current_state[i].keys():
                        if isinstance(current_state[i][j], (numbers.Number, list)) and isinstance(state[i][j], (numbers.Number, list)):
                            current_state[i][j] += state[i][j]
        return current_state

    def calc_delta(self, current_state, new_state):
                new_state = copy.deepcopy(new_state)
                for i in new_state.keys():
                    for j in new_state[i].keys():
                        if isinstance(new_state[i][j], numbers.Number):
                            new_state[i][j] -= current_state[i][j]
                return new_state

    def compose_instructions(self, agents_info, delta):
            _delta = copy.deepcopy(delta)
            _agents_info = copy.deepcopy(agents_info)
            for aid, state in _agents_info.items():
                for i in _delta.keys():
                        max_inc = state[i]['max'] - state[i]['level']
                        max_dec = state[i]['level'] - state[i]['min']
                        if _delta[i]['level'] > ZERO:
                            if _delta[i]['level'] <= max_inc:
                                state[i]['level'] += _delta[i]['level']
                                _delta[i]['level'] = 0
                            else:
                                state[i]['level'] += max_inc
                                _delta[i]['level'] -= max_inc
                        elif _delta[i]['level'] <= ZERO:            
                            if abs(_delta[i]['level']) <= max_dec:
                                state[i]['level'] += _delta[i]['level']
                                _delta[i]['level'] = 0
                            else:
                                state[i]['level'] -= max_dec
                                _delta[i]['level'] += max_dec
                        
                        state[i]['delta'] = state[i]['level'] - agents_info[aid][i]['level']
            return _agents_info, _delta

    def adjust_instruction(self, current_state, new_state):
            _current_state = copy.deepcopy(current_state)
            key = 'consumption'
            _current_state[key]['delta'] = current_state[key]['level'] - new_state[key]['level']
            _current_state[key]['level'] = new_state[key]['level']
            key = 'production'
            _current_state[key]['delta'] = current_state[key]['level'] - new_state[key]['level']
            _current_state[key]['level'] = new_state[key]['level']
            return _current_state

    def compute_instructions(self, current_state, **kwargs):
        verbose = kwargs.get('verbose', 0)

        def compute_balance(external_network_state,
                                        cells_aggregated_state):
                                
                    old_cells_aggregated_state = copy.deepcopy(cells_aggregated_state)
                    old_external_network_state = copy.deepcopy(external_network_state)
                    cell_balance = cells_aggregated_state['production']['level'] - cells_aggregated_state['consumption']['level']
                    cells_aggregated_state['production']['level'] = np.clip(cells_aggregated_state['production']['level'], 
                                                                                cells_aggregated_state['production']['min'], 
                                                                                cells_aggregated_state['production']['max'])
                    cells_aggregated_state['consumption']['level'] = np.clip(cells_aggregated_state['consumption']['level'], 
                                                                                cells_aggregated_state['consumption']['min'], 
                                                                                cells_aggregated_state['consumption']['max'])
                    cell_expected_balance = cells_aggregated_state['production']['level'] - cells_aggregated_state['consumption']['level']

                    external_network_min = external_network_state['consumption']['max'] * (-1)
                    external_network_max = external_network_state['production']['max']
                    external_network_default_balance = np.clip(external_network_state['production']['level'] - external_network_state['consumption']['level'],# - cell_balance,
                                                                external_network_min,
                                                                external_network_max)
                    if verbose >= 0:
                        print(highlight('cells balance:', 'red'), cell_balance)
                        print(highlight('cells expected balance:', 'red'), cell_expected_balance)
                        print(highlight('external network default balance:', 'blue'), external_network_default_balance)

                    cell_inc_production = cells_aggregated_state['production']['max'] - cells_aggregated_state['production']['level']
                    if cell_inc_production < ZERO:
                        cell_inc_production = 0
                    cell_dec_production = cells_aggregated_state['production']['level'] - cells_aggregated_state['production']['min']
                    if cell_dec_production < ZERO:
                        cell_dec_production = 0
                    cell_dec_consumption = cells_aggregated_state['consumption']['level'] - cells_aggregated_state['consumption']['min']
                    if cell_dec_consumption < ZERO:
                        cell_dec_consumption = 0
                    cell_inc_consumption = cells_aggregated_state['consumption']['max'] - cells_aggregated_state['consumption']['level']
                    if cell_inc_consumption < ZERO:
                        cell_inc_consumption = 0

                    grid_inc = external_network_max - external_network_default_balance
                    grid_dec = external_network_default_balance - external_network_min

                    if cell_expected_balance < ZERO:
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
                    elif cell_expected_balance > ZERO:
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

                    cells_aggregated_state['production']['level'] = cells_aggregated_state['production']['level'] + cell_inc_production - cell_dec_production
                    cells_aggregated_state['consumption']['level'] = cells_aggregated_state['consumption']['level'] + cell_inc_consumption - cell_dec_consumption     
                    new_cell_balance = cells_aggregated_state['production']['level'] - cells_aggregated_state['consumption']['level']
                    
                    new_external_network_balance = external_network_default_balance + grid_inc - grid_dec
                    if new_external_network_balance > ZERO:
                        external_network_state['production']['level'] = new_external_network_balance
                        external_network_state['consumption']['level'] = 0
                    else:
                        external_network_state['production']['level'] = 0
                        external_network_state['consumption']['level'] = abs(new_external_network_balance)      

                    cells_aggregated_delta = self.calc_delta(old_cells_aggregated_state, cells_aggregated_state)
                    external_network_delta = self.calc_delta(old_external_network_state, external_network_state)

                    #print('new_cells_aggregated_state', cells_aggregated_state)
                    #print('new_external_network_state', external_network_state)
                    #print('cells_aggregated_delta', cells_aggregated_delta)
                    #print('external_network_delta', external_network_delta)

                    if verbose >= 0:
                        print(highlight('new cells balance:', 'red'), new_cell_balance)
                        print(highlight('external network balance:', 'blue'), new_external_network_balance)

                    ok = abs(new_cell_balance - cell_expected_balance) < ZERO
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
            delta = self.calc_delta(current_state, instruction)
            instructions, remains = self.compose_instructions(requested_states, delta)
        else:
            cells_aggregated_state = self.aggregation_method(None, None, requested_states)
            ok, _, cells_delta, state, _ = compute_balance(current_state, cells_aggregated_state)
            instructions, remains = self.compose_instructions(requested_states, cells_delta)

        return ok, instructions, state

    def execute_method(self, aeid, aid, instruction, current_state, requested_states, current_time, first_time_step, **kwargs):
        ok = True   
        between = kwargs.get('between_cells', 'default')
        within = kwargs.get('within_cell', 'default')

        if not len(requested_states) == 0: # not a leaf agent
            if not aeid == 'MosaikAgent':
                if within == 'default':
                    ok, instructions, state = self.compute_instructions(instruction=instruction, 
                                                                        current_state=current_state,
                                                                requested_states=requested_states, **kwargs)
                elif callable(within) or getattr(within, 'execute', False):
                    within = getattr(within, 'execute', within)
                    if first_time_step:
                        agents = []
                        flexibility = []
                        zero_flexibility = []
                        total_fixed_values = 0
                        for agent, old_state in requested_states.items():
                                agents.append(agent)
                                if abs(old_state['production']['max'] - old_state['production']['min']) > ZERO:
                                    zero_flexibility.append(None)
                                    flexibility.append({'flex_max_power': [old_state['production']['max']],
                                                        'flex_min_power': [old_state['production']['min']]})
                                else:
                                    zero_flexibility.append([old_state['production']['min']])
                                    total_fixed_values += old_state['production']['min']

                        target_schedule = [instruction['production']['level'] - total_fixed_values]
                        if kwargs.get('verbose', 0) >= 0:
                            print('target_schedule:', target_schedule)
                            print('agents:', agents)
                            print('zero_flexibility:', zero_flexibility)
                            print('flexibility:', flexibility)

                        if (len(requested_states.keys()) > 1) and (len([i for i in zero_flexibility if i == None]) > 1): # if there is only agent, no algorithm needed
                            schedules = within(target_schedule=target_schedule,
                                                            flexibility=flexibility, **kwargs)
                        else:
                            clip_max = [sum(i['flex_max_power']) for i in flexibility]
                            clip_min = [sum(i['flex_min_power']) for i in flexibility]
                            schedules = [np.clip(target_schedule, clip_min, clip_max)]

                        self.cache.setdefault('within_cell', {})
                        self.cache['within_cell'].setdefault(aeid, {})
                        schedules = iter(schedules)
                        for agent, i in zip(agents, zero_flexibility):
                            if i == None:
                                self.cache['within_cell'][aeid][agent] = next(schedules)
                            else:
                                self.cache['within_cell'][aeid][agent] = i
                        if kwargs.get('verbose', 0) >= 0:
                            print('schedules:', self.cache['within_cell'][aeid])
                        
                    instructions = {}
                    for agent, old_state in requested_states.items():
                        new_state = copy.deepcopy(old_state)
                        schedule = self.cache['within_cell'][aeid][agent]
                        new_state['production']['level'] = schedule[0]
                        instructions[agent] = self.adjust_instruction(old_state, new_state)

                else:
                    print("The within-cell communication algorithm can't be executed.")
                    sys.exit()

                state = copy.deepcopy(self.config['state_dict'])
            else:
                if between == 'default':
                    ok, instructions, state = self.compute_instructions(instruction=instruction, 
                                                                        current_state=current_state,
                                                                requested_states=requested_states, **kwargs)
                elif callable(between) or getattr(between, 'execute', False):
                    between = getattr(between, 'execute', between)
                    state = copy.deepcopy(current_state)
                    if first_time_step:
                        agents = []
                        flexibility = []
                        total_consumption = 0
                        for agent, old_state in requested_states.items():
                            agents.append(agent)
                            flexibility.append({'flex_max_power': [old_state['production']['max']],
                                                'flex_min_power': [old_state['production']['min']]})
                            total_consumption += old_state['consumption']['level']
                            
                        flexibility += [{'flex_max_power': [current_state['production']['max']],
                                        'flex_min_power': [current_state['production']['min']]}]

                        target_schedule = [total_consumption]
                        if kwargs.get('verbose', 0) >= 0:
                            print('target_schedule:', target_schedule)
                            print('agents:', agents)
                            print('flexibility:', flexibility)

                        schedules = between(target_schedule=target_schedule,
                                                        flexibility=flexibility, **kwargs)
                        
                        self.cache.setdefault('between_cells', {})
                        self.cache['between_cells'].setdefault(aeid, {})
                        for agent, schedule in zip(agents, schedules):
                            self.cache['between_cells'][aeid][agent] = schedule

                        if kwargs.get('verbose', 0) >= 0:
                            print('schedules:', self.cache['between_cells'][aeid])
                    
                    instructions = {}
                    for agent, old_state in requested_states.items():
                        new_state = copy.deepcopy(old_state)
                        schedule = self.cache['between_cells'][aeid][agent]
                        new_state['production']['level'] = schedule[0]
                        instructions[agent] = self.adjust_instruction(old_state, new_state)

                else:
                    print("The between-cells communication algorithm can't be executed.")
                    sys.exit()

        else:
            instructions = {aid : instruction}
            state = instruction

        return ok, instructions, state