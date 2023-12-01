import asyncio

import pytest
from mango.core.container import Container
from src.mas.mosaik import MosaikAgent
from src.mas.controller import Controller
from src.mas.wecs import WecsAgent
from src.mas.agent_messages import CurrentPMaxMessage, create_msg_content


@pytest.mark.asyncio
@pytest.fixture
async def controller_container():
    c = await Container.factory(addr=('127.0.0.1', 5555))
    yield c
    await c.shutdown()

@pytest.mark.asyncio
@pytest.fixture
async def agent_container():
    c = await Container.factory(addr=('127.0.0.1', 5556))
    yield c
    await c.shutdown()

@pytest.mark.asyncio
@pytest.fixture
async def mosaik_agent_container():
    c = await Container.factory(addr=('127.0.0.1', 5557))
    yield c
    await c.shutdown()


@pytest.mark.asyncio
async def test_init(mosaik_agent_container):
    m = MosaikAgent(mosaik_agent_container)
    await m.shutdown()


@pytest.mark.asyncio
async def test_reset(mosaik_agent_container):
    m = MosaikAgent(mosaik_agent_container)
    m.agents = {'eid1': (('localhost', 5555), 'agent0'),
                'eid2': (('localhost', 5555), 'agent1')}
    m.reset()
    assert len(m._p_max) == 2
    assert isinstance(m._p_max['eid1'], asyncio.Future) and not m._p_max['eid1'].done()
    assert isinstance(m._p_max['eid2'], asyncio.Future) and not m._p_max['eid2'].done()


@pytest.mark.asyncio
async def test_handle_message(mosaik_agent_container):
    m = MosaikAgent(mosaik_agent_container)
    m.agents = {'eid1': (('localhost', 5555), 'agent0'),
                'eid2': (('localhost', 5555), 'agent1')}
    m.reset()
    m.handle_msg(content=create_msg_content(CurrentPMaxMessage, p_max=10), meta={'conversation_id': 'eid1'})
    m.handle_msg(content=create_msg_content(CurrentPMaxMessage, p_max=20), meta={'conversation_id': 'eid2'})
    assert m._p_max['eid1'].done() and m._p_max['eid1'].result() == 10
    assert m._p_max['eid2'].done() and m._p_max['eid2'].result() == 20


@pytest.mark.asyncio
async def test_get_p_max(controller_container, agent_container, mosaik_agent_container):
    # we need to create a controller as it is needed in the constructor of wecs
    controller = Controller(controller_container, max_windpark_feedin=0)
    controller_addr = controller_container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent_i = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 10000})

    # create another wecsAgent
    wecs_agent_j = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})
    # set p max values
    wecs_agent_i.p_max = 5000
    wecs_agent_j.p_max = 10000

    m = MosaikAgent(mosaik_agent_container)
    m.agents = {
        'wecs_agent_i': (agent_container.addr, wecs_agent_i.aid),
        'wecs_agent_j': (agent_container.addr, wecs_agent_j.aid),
    }
    m.reset()
    p_max_dict = await m.get_p_max()
    assert len(p_max_dict.keys()) == 2
    assert 'wecs_agent_i' in p_max_dict.keys() and 'wecs_agent_j' in p_max_dict.keys()
    assert p_max_dict['wecs_agent_i'] == 5000
    assert p_max_dict['wecs_agent_j'] == 10000


@pytest.mark.asyncio
async def test_set_p_max(controller_container, agent_container, mosaik_agent_container):
    # we need to create a controller as it is needed in the constructor of wecs
    controller = Controller(controller_container, max_windpark_feedin=30000)
    controller_addr = controller_container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent_i = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 10000})

    # create another wecsAgent
    wecs_agent_j = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})

    # set current p
    wecs_agent_i.current_p = 5000
    wecs_agent_j.current_p = 10000

    m = MosaikAgent(mosaik_agent_container)
    m.agents = {
        'wecs_agent_i': (agent_container.addr, wecs_agent_i.aid),
        'wecs_agent_j': (agent_container.addr, wecs_agent_j.aid),
    }
    m.controller = (controller_addr, controller_id)
    m.reset()
    await asyncio.wait_for(m.trigger_control_cycle(), timeout=2)

    p_max_dict = await asyncio.wait_for(m.get_p_max(), timeout=2)
    assert len(p_max_dict.keys()) == 2
    assert 'wecs_agent_i' in p_max_dict.keys() and 'wecs_agent_j' in p_max_dict.keys()
    assert p_max_dict['wecs_agent_i'] == 10000
    assert p_max_dict['wecs_agent_j'] == 20000

@pytest.mark.asyncio
async def test_trigger_control_cycle(controller_container, agent_container, mosaik_agent_container):
    # we need to create a controller as it is needed in the constructor of wecs
    controller = Controller(controller_container, max_windpark_feedin=30000)
    controller_addr = controller_container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent_i = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 10000})

    # create another wecsAgent
    wecs_agent_j = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})

    # set current p
    wecs_agent_i.current_p = 5000
    wecs_agent_j.current_p = 10000

    m = MosaikAgent(mosaik_agent_container)
    m.agents = {
        'wecs_agent_i': (agent_container.addr, wecs_agent_i.aid),
        'wecs_agent_j': (agent_container.addr, wecs_agent_j.aid),
    }
    m.controller = (controller_addr, controller_id)
    m.reset()
    await m.trigger_control_cycle()

    p_max_dict = await m.get_p_max()
    assert len(p_max_dict.keys()) == 2
    assert 'wecs_agent_i' in p_max_dict.keys() and 'wecs_agent_j' in p_max_dict.keys()
    assert p_max_dict['wecs_agent_i'] == 10000
    assert p_max_dict['wecs_agent_j'] == 20000


@pytest.mark.asyncio
async def test_update_state(agent_container, mosaik_agent_container):
    # we don't need to create a controller for this test

    # create a wecsAgent
    wecs_agent_i = WecsAgent(container=agent_container, controller_addr=None,
                             controller_id=None, model_conf={'P_rated': 10000})

    # create another wecsAgent
    wecs_agent_j = WecsAgent(container=agent_container, controller_addr=None,
                             controller_id=None, model_conf={'P_rated': 20000})

    m = MosaikAgent(mosaik_agent_container)
    m.agents = {
        'wecs_agent_i': (agent_container.addr, wecs_agent_i.aid),
        'wecs_agent_j': (agent_container.addr, wecs_agent_j.aid),
    }
    m.reset()
    new_state_dict = {'wecs_agent_i': {'P': 1000},
                      'wecs_agent_j': {'P': 2000}}
    await m.update_agents(new_state_dict)

    assert wecs_agent_i.current_p == 1000
    assert wecs_agent_j.current_p == 2000


@pytest.mark.asyncio
async def test_step(controller_container, agent_container, mosaik_agent_container):

    # max_windpark_feedin is exceeded
    inputs_1 = {
        'wecs_agent_i': {
            'P':  10000.0
        },
        'wecs_agent_j': {
            'P': 20000.0
        }
    }

    expected_outputs_1 = {
        'wecs_agent_i': 5000.0,
        'wecs_agent_j': 10000.0,
    }

    # max_windpark_feedin is not exceeded
    inputs_2 = {
        'wecs_agent_i': {
            'P': 1000.0
        },
        'wecs_agent_j': {
            'P': 4000.0
        }
    }

    expected_outputs_2 = {
        'wecs_agent_i': 10000.0,
        'wecs_agent_j': 20000.0,
    }

    # create controller
    controller = Controller(controller_container, max_windpark_feedin=15000)
    controller_addr = controller_container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent_i = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 10000})

    # create another wecsAgent
    wecs_agent_j = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})

    # create a MosaikAgent
    m = MosaikAgent(mosaik_agent_container)
    m.agents = {
        'wecs_agent_i': (agent_container.addr, wecs_agent_i.aid),
        'wecs_agent_j': (agent_container.addr, wecs_agent_j.aid),
    }
    m.controller = (controller_addr, controller_id)

    outputs_1 = await m.step(inputs_1)
    outputs_2 = await m.step(inputs_2)

    assert outputs_1 == expected_outputs_1
    assert outputs_2 == expected_outputs_2


