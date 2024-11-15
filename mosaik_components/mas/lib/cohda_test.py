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
    cohda = COHDA(base_port=10000, muted=True)

    target_schedule = [0.5, 2.0, 5.0]
    flex = {'flex_max_power': [3.0, 3.0, 3.0],
            'flex_min_power': [0.1, 0.2, 0.3],
            }
    print('n_agents:', n_agents)
    print('target_schedule:', target_schedule)
    print('flexibility:', flex)
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)])
    print('schedules:', schedules)

    time.sleep(1)
    print()

    n_agents = 3
    print('n_agents:', n_agents)
    target_schedule = [0.1, 0.0, 2.0]
    flex = {'flex_max_power': [0.0, 1.0, 1.0],
            'flex_min_power': [0.0, 0.0, 0.0],
            }
    print('target_schedule:', target_schedule)
    print('flexibility:', flex)
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)])
    print('schedules:', schedules)
    last_schedules = schedules

    print()
    print('run again with the same input, to test caching..')
    schedules = cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)])
    print('schedules:', schedules)
    assert schedules == last_schedules

if __name__ == '__main__':
    test_cohda()

