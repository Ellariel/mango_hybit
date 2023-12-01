import pytest
import asyncio
from mango.core.container import Container
from src.mas.controller import Controller
from src.mas.wecs import WecsAgent
from src.mas.agent_messages import TriggerControlActions

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
async def controller(controller_container: Container):
    controller = Controller(controller_container, max_windpark_feedin=10000)
    yield controller
    await controller.shutdown()


@pytest.mark.asyncio
async def test_registration(controller, agent_container):
    controller_addr = controller._container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent = await WecsAgent.create(container=agent_container, controller_addr=controller_addr,
                           controller_id=controller_id, model_conf={})

    # make sure registration is complete
    assert controller.wecs == [(agent_container.addr, wecs_agent.aid)]


@pytest.mark.asyncio
async def test_control_cycle(controller: Controller):
    controller_addr = controller._container.addr
    controller_id = controller.aid

    # create a wecsAgent
    wecs_agent_i = await WecsAgent.create(container=controller._container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})

    wecs_agent_i.current_p = 2000

    # p_max should initially be None
    assert wecs_agent_i.p_max is None

    controller.handle_msg(content=TriggerControlActions(), meta={})
    await controller.cycle_done
    assert wecs_agent_i.p_max == 20000

    wecs_agent_j = await WecsAgent.create(container=controller._container, controller_addr=controller_addr,
                             controller_id=controller_id, model_conf={'P_rated': 20000})
    wecs_agent_j.current_p = 18000
    controller.handle_msg(content=TriggerControlActions(), meta={})
    await controller.cycle_done
    assert wecs_agent_i.p_max == int(1000)
    assert wecs_agent_j.p_max == int(9000)
