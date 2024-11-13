import logging
import time
import nest_asyncio
from cohda import COHDA

def test_cohda():
    """
    Run test.
    """
    nest_asyncio.apply()
    logging.basicConfig(level=logging.INFO)

    n_agents = 5
    cohda = COHDA(base_port=10000, verbose=False)

    target_schedule = [0.5, 2.0, 5.0]
    flexibility = {'flex_max_power': [3.0, 3.0, 3.0],
            'flex_min_power': [0.1, 0.2, 0.3],
            }
    print('n_agents:', n_agents)
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    assert len(schedules['Agent_0']['FlexSchedules']) == 3

    time.sleep(1)
    print()

    n_agents = 3
    print('n_agents:', n_agents)
    target_schedule = [0.1, 0.0, 2.0]
    flexibility = {'flex_max_power': [0.0, 1.0, 1.0],
            'flex_min_power': [0.0, 0.0, 0.0],
            }
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    last_schedules = schedules

    print()
    print('run again with the same input, to test caching..')
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_agents)])
    print('schedules:', schedules)
    assert schedules == last_schedules

if __name__ == '__main__':
    test_cohda()

