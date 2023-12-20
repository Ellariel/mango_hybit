
"""


"""

import asyncio
import logging
import mosaik_api
#import mosaik_api_v3 as mosaik_api
import mango
from agent_messages import *
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import nest_asyncio
nest_asyncio.apply()

class Agent(mango.Agent):
    def __init__(self, container, controller, initial_states=[]):
        super().__init__(container)
        self.container = container
        self.controller = controller
        self.states = initial_states
        self.connected_agents = []
        self._registration_confirmed = asyncio.Future()
        self._instruction_confirmed = {}
        self._requested_info = {}
        # print(f"Hello world! I am an Agent. My id is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """

        content = read_msg_content(content)
        print(f"{self.aid} received: {content}")
        if isinstance(content, RegisterRequestMessage):
            self.register_agent(content, meta)
        elif isinstance(content, RegisterConfirmMessage):
            self._registration_confirmed.set_result(True)
        elif isinstance(content, UpdateStateMessage):
            self.update_state(content, meta)
        elif isinstance(content, RequestInformationMessage):
            self.schedule_instant_task(self.send_info(content, meta))
        elif isinstance(content, AnswerInformationMessage):
            self._requested_info[meta['conversation_id']].set_result(content.info)
        elif isinstance(content, TriggerControlActions):
            self.schedule_instant_task(self.perform_control_actions(meta))
        elif isinstance(content, BroadcastInstructionsMessage):
            self.schedule_instant_task(self.execute_instructions(content, meta))
        elif isinstance(content, InstructionsConfirmMessage):
            self._instruction_confirmed[meta['conversation_id']].set_result(content.instructions)
        else:
            pass

    async def register(self):
        """
        Schedule a task that sends a RegisterRequestMessage to the controller agent and wait for the confirmation.
        This should be called once an agent is initialized.
        """
        msg_content = create_msg_content(RegisterRequestMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(
            coroutine=self.container.send_acl_message(
                content=msg_content,
                receiver_addr=self.controller[0], 
                receiver_id=self.controller[1],
                acl_metadata={}
            ))
        await asyncio.wait_for(self._registration_confirmed, timeout=3)

    def register_agent(self, content: RegisterRequestMessage, meta):
        """
        Registered an agent
        :param content: Register message
        :param meta: The meta information dict
        """
        addr = (content.host, content.port)
        aid = content.aid
        self.connected_agents.append((addr, aid))
        
        msg_content = create_msg_content(RegisterConfirmMessage,
                                         aid=self.aid, host=self.container.addr[0], port=self.container.addr[1])
        self.schedule_instant_task(self.container.send_acl_message(
            receiver_addr=addr,
            receiver_id=aid,
            content=msg_content,
            acl_metadata={},
        ))

    def update_state(self, content, meta):
        """
        Update the current state of connected Entity
        :param content: The state content including the current state coming from mosaik
        :param meta: the meta information dict
        """
        # state={'P[MW]': {'Grid-0.Gen-0': 1.0}}
        for k, v in content.state.items():
            for eid, state in v.items():
                if 'Load' in eid and state > 0: # check consumtion/production
                    v[eid] *= -1
                elif ('Gen' in eid or 'Wecs' in eid) and state < 0:
                    v[eid] *= -1
                else:
                    pass

        self.states.append(content.state) # store current state

        msg_content = create_msg_content(UpdateConfirmMessage)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))

    async def send_info(self, content, meta):
        """
        Send requested information (e.g. of the current state or any calculated data of connected Entity)
        :param content: The content including the information requested
        :param meta: The meta information dict
        """

        data = {}
        if len(self.states):
            for v in self.states[-1].keys():
                values = [j for i in self.states if v in i for j in i[v].values()]     
                data[v] = {'min' : min(values),
                           'max' : max(values),
                           'cur' : values[-1]}
        data = {self.aid : {'flexibility' : data}}
                
        # Request information
        if len(self.connected_agents):
            self._requested_info = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(RequestInformationMessage, info={}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._requested_info = await asyncio.gather(*[fut for fut in self._requested_info.values()])
            self._requested_info = {k : v for i in self._requested_info for k, v in i.items()}
            data.update(self._requested_info)

        msg_content = create_msg_content(AnswerInformationMessage, info=data)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))        

    async def execute_instructions(self, content, meta):
        """
        Execute received instructions and sent the confirmation
        :param content: The content including instructions to execute
        :param meta: The meta information dict
        """

        # Do execute


        if len(self.connected_agents):
            # Broadcast instructions
            self._instruction_confirmed = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(BroadcastInstructionsMessage, instructions={self.aid : aid}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._instruction_confirmed = await asyncio.gather(*[fut for fut in self._instruction_confirmed.values()])

        msg_content = create_msg_content(InstructionsConfirmMessage, instructions={'aid': self.aid,
                                                                                   'instructions': content.instructions})
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            )) 

    async def perform_control_actions(self, meta):
        """
        """
        # Request information if there are connected agents
        if len(self.connected_agents):
            self._requested_info = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(RequestInformationMessage, info={}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._requested_info = await asyncio.gather(*[fut for fut in self._requested_info.values()])
            self._requested_info = {k : v for i in self._requested_info for k, v in i.items()}

            print('ACTION')
            print('requested_info', self._requested_info)
            #requested_info requested_info {'agent2': {'flexibility': {'P[MW]': {'min': -1.0, 'max': 1.0, 'cur': 0.00404922087048476}}}, 
            
            # calc flexibility
            #for state in self._requested_info:
            #    for k, v in state.items():
            #        v['flexibility'] = (v['cur'] - v['min']) / (v['max'] - v['min']) if (v['max'] - v['min']) != 0 else 0 

            # calc balance
            #p_in_flow = 0
            #p_out_flow = 0
            #for state in self._requested_info:
            #    for k, v in state.items():
            #        if 'P' in k:
            #            if v['cur'] < 0:
            #                p_out_flow += v['cur']
            #            else:
            #                p_in_flow += v['cur']
            #print(f"p_in_flow: {p_in_flow}, p_out_flow: {p_out_flow}, balance: {p_in_flow + p_out_flow}")
            
            #print(self._requested_info)

            # Send instructions
            print('SEND INSTRUCTIONS')
            self._instruction_confirmed = {aid: asyncio.Future() for _, aid in self.connected_agents}
            futs = [self.schedule_instant_task(self.container.send_acl_message(
                            receiver_addr=addr,
                            receiver_id=aid,
                            content=create_msg_content(BroadcastInstructionsMessage, instructions={self.aid : aid}),
                            acl_metadata={'conversation_id': aid,
                                        'sender_id': self.aid},
                    ))
                    for addr, aid in self.connected_agents]
            await asyncio.gather(*futs)
            self._instruction_confirmed = await asyncio.gather(*[fut for fut in self._instruction_confirmed.values()])

            print('INSTRUCTIONS CONFIRMED')

        # confirm if sender_id is provided
        msg_content = create_msg_content(ControlActionsDone, info=self._instruction_confirmed)
        if 'sender_id' in meta.keys():
            conv_id = meta.get('conversation_id', None)
            self.schedule_instant_task(self.container.send_acl_message(
                content=msg_content,
                receiver_addr=meta['sender_addr'],
                receiver_id=meta['sender_id'],
                acl_metadata={'conversation_id': conv_id},
            ))  

class MosaikAgent(mango.Agent):
    def __init__(self, container):
        super().__init__(container)
        self.container = container
        self._updates_received = {}
        self._controllers_done = {}
        self._all_agents = {}
        self._controllers = {}
        #print(f"Hello world! I am MosaikAgent. My id is {self.aid}")

    def handle_message(self, content, meta):
        """
        Handling all incoming messages of the agent
        :param content: The message content
        :param meta: All meta information in a dict.
        Possible keys are e.g. sender_addr, sender_id, conversation_id...
        """
        content = read_msg_content(content)
        print(f"{self.aid} received: {content}")
        if isinstance(content, UpdateConfirmMessage):
            self._updates_received[meta['conversation_id']].set_result(True)
        elif isinstance(content, ControlActionsDone):
            self._controllers_done[meta['conversation_id']].set_result(content.info)
        else:
            pass

    async def update_agents(self, data):
        """
        Update the agents with new data from mosaik.
        """
        self._updates_received = {aid: asyncio.Future() for aid in data.keys()}
        #print(data)
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=self._all_agents[mosaik_eid][0],
                    receiver_id=self._all_agents[mosaik_eid][1],
                    content=create_msg_content(UpdateStateMessage, state=input_data),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, input_data in data.items()]
        await asyncio.gather(*futs)
        await asyncio.gather(*[fut for fut in self._updates_received.values()])

    async def trigger_control_cycle(self):
        """
        Trigger control cycle after all the agents got their updates from Mosaik.
        """
        self._controllers_done = {aid: asyncio.Future() for aid in self._controllers.keys()}
        futs = [self.schedule_instant_task(self.container.send_acl_message(
                    receiver_addr=controller[0],
                    receiver_id=controller[1],
                    content=create_msg_content(TriggerControlActions),
                    acl_metadata={'conversation_id': mosaik_eid,
                                  'sender_id': self.aid},
            ))
            for mosaik_eid, controller in self._controllers.items()]
        await asyncio.gather(*futs)
        self._controllers_done = await asyncio.gather(*[fut for fut in self._controllers_done.values()])
        return {eid : info['instructions'] for eid, a in self._all_agents.items() 
                                                for info in self._controllers_done[0] 
                                                    if a[1] == info['aid']}

    async def loop_step(self, inputs):
            """
            This will be called from the mosaik api once per step.

            :param inputs: the input dict from mosaik: {eid_1: {'P': p_value}, eid_2: {'P': p_value}...}
            :return: the output dict: {eid_1: p_max_value, eid_2: p_max_value}
            """
            # 1. update state of agents
            await self.update_agents(inputs)
            # 2. trigger control actions
            return await self.trigger_control_cycle()

#logger = logging.getLogger('mosaik_agents')
def main():
    """Run the multi-agent system."""
    #logging.basicConfig(level=logging.INFO)
    return mosaik_api.start_simulation(MosaikAgents())

# The simulator meta data that we return in "init()":
META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'MosaikAgents': {
            'public': True,
            'any_inputs': True,
            'params': ['controller', 'initial_states'],
            'attrs': [],
        },
    },
}

class MosaikAgents(mosaik_api.Simulator):
    """
    Interface to mosaik.
    """

    def __init__(self):
        super().__init__(META)
        self.step_size = 60 * 15  # We have a step size of 15 minutes specified in seconds:
        self.host = 'localhost'
        self.port = 5678
        # Set by "run()":
        self.mosaik = None  # Proxy object for mosaik
        # Set in "init()"
        self.sid = None  # Mosaik simulator ID
        self.loop = None  # Mango agents loop
        self.main_container = None # Mango container
        self.mosaik_agent = None # Mosaik Mango Agent
        # Updated in "create()"
        #self.cell_agents = {}  # eid : ((host,port), aid)
        self.controllers = {}  # eid : ((host,port), aid)
        # Set/updated in "setup_done()"
        self.all_agents = {} # contains agents + controllers for technical tasks
        self.entities = {}  # agent_id: unit_id
        #self.loop = asyncio.get_running_loop()

    def init(self, sid, time_resolution=1., **sim_params):
        self.sid = sid
        self.loop = asyncio.get_event_loop()
        self.main_container = self.loop.run_until_complete(self._create_container(self.host, self.port))
        self.mosaik_agent = self.loop.run_until_complete(self._create_mosaik_agent(self.main_container))
        return META

    async def _create_container(self, host, port):
        return await mango.create_container(addr=(host, port))
    
    async def _create_mosaik_agent(self, container):
        return MosaikAgent(container)

    async def _create_agent(self, container, controller, initial_states=[]):
        if controller == None:
            agent = Agent(container, None, initial_states)
        else:
            agent = Agent(container, self.all_agents[controller], initial_states)
            await agent.register()
        return agent

    def create(self, num, model, **model_conf):
        """
        Create *num* instances of *model* and return a list of entity dicts
        to mosaik.
        """
        assert model in META['models']
        # Get the number of agents created so far and count from this number
        # when creating new entity IDs:
        entities = []
        '''
        if model_conf['controller'] == None:
            n_agents = len(self.controllers)
            for i in range(n_agents, n_agents + num):
                eid = 'ControllerAgent_%s' % i
                agent = self.loop.run_until_complete(self._create_agent(self.main_container, 
                                                                        None, 
                                                                        model_conf['initial_states']))
                self.controllers[eid] = (self.main_container.addr, agent.aid)
                entities.append({'eid': eid, 'type': model})
        else:
        '''
        n_agents = len(self.all_agents)
        for i in range(n_agents, n_agents + num):
                eid = 'Agent_%s' % i
                agent = self.loop.run_until_complete(self._create_agent(self.main_container, 
                                                                        model_conf['controller'], 
                                                                        model_conf['initial_states']))
                self.all_agents[eid] = (self.main_container.addr, agent.aid)
                if model_conf['controller'] == None:
                    self.controllers[eid] = (self.main_container.addr, agent.aid)
                entities.append({'eid': eid, 'type': model})                  
            # as the event loop is not running here, we have to create the agents via loop.run_unti_complete.
            # That means however, that the agents are not able to receive or send messages unless the loop is
            # triggered from this module.
        return entities

    def setup_done(self):
        """
        Get the entities that our agents are connected to once the scenario
        setup is done.
        """
        #self.all_agents = self.cell_agents.copy()
        #self.all_agents.update(self.controllers)
        self.mosaik_agent._all_agents = self.all_agents
        self.mosaik_agent._controllers = self.controllers

        #print(self.controllers)

        full_ids = ['%s.%s' % (self.sid, eid) for eid in self.all_agents.keys()]
        relations = yield self.mosaik.get_related_entities(full_ids)
        #print(full_ids)
        print('relations:', relations)
        for full_aid, units in relations.items():
            if len(units):
                # We should be connected to at least one entity
                #assert len(units) >= 1
                uid, _ = units.popitem()
                # Create a mapping "agent ID -> unit ID"
                aid = full_aid.split('.')[-1]
                self.entities[aid] = uid
        print('entities:', self.entities)

    def finalize(self):
        self.loop.run_until_complete(self._shutdown(self.main_container))

    @staticmethod
    async def _shutdown(*args):
        futs = []
        for arg in args:
            futs.append(arg.shutdown())
        print('Going to shutdown agents and container... ', end='')
        await asyncio.gather(*futs)
        print('done.')

    def step(self, time, inputs, max_advance):
        """Send the inputs of the controlled unites to our agents and get new
        set-points for these units from the agents.

        This method will run for at most "step_size" seconds, even if the
        agents need longer to do their calculations.  They will then continue
        to do stuff in the background, while this method returns and allows
        mosaik to continue the simulation.

        """
        print('inputs:', inputs)
        # trigger the loop to enable agents to send / receive messages via run_until_complete
        output_dict = self.loop.run_until_complete(self.mosaik_agent.loop_step(inputs=inputs))


        print('output', output_dict)
        #print(self.all_agents)

        # Make "set_data()" call back to mosaik to send the set-points:
        #inputs = {aid: {self.entities[aid]: v}
        #          for aid, v in output_dict.items()}
        
        inputs = {}
        
        yield self.mosaik.set_data(inputs)

        return time + self.step_size
    
    def get_data(self, outputs):
        # we are going to send the data asynchronously via set_data, hence we do not need to implement get_data()
        pass

if __name__ == '__main__':
    main()


