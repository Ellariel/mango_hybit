import time
import numpy as np
import nest_asyncio
from massca.lib.cohda import COHDA

def test_cohda():
    """
    Run test.
    """
    nest_asyncio.apply()

    n_agents = 5
    cohda = COHDA(base_port=10000, verbose=False)
    target_schedule = [1.5, 5.0, 15.0]
    flexibility = {'flex_max_power': [3.0, 3.0, 3.0],
            'flex_min_power': [0.0, 0.1, 0.3]}
    print('n_agents:', n_agents)
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = cohda.execute(seed=13, target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    print('solution:', schedules.sum(0))
    rmse = np.sqrt(np.sum((target_schedule - schedules.sum(0))**2)/len(target_schedule))
    print('RMSE:', rmse)
    assert len(schedules) == n_agents
    assert len(schedules[0]) == len(target_schedule)
    assert rmse < max(target_schedule)

    time.sleep(1)
    print()

    n_agents = 3
    print('n_agents:', n_agents)
    target_schedule = [0.1, 0.0, 2.0]
    flexibility = {'flex_max_power': [0.0, 1.0, 1.0],
            'flex_min_power': [0.0, 0.0, 0.0]}
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = cohda.execute(seed=13, target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    print('solution:', schedules.sum(0))
    rmse = np.sqrt(np.sum((target_schedule - schedules.sum(0))**2)/len(target_schedule))
    print('RMSE:', rmse)
    last_schedules = schedules
    assert rmse < max(target_schedule)

    print()
    print('run again with the same input, to test caching..')
    schedules = cohda.execute(seed=13, target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    assert (schedules == last_schedules).all()

if __name__ == '__main__':
    test_cohda()

