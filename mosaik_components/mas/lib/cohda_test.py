import logging
import time
import nest_asyncio
from cohda import COHDA

def test():
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
    print('schedules:', cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)]))
    
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
    print('schedules:', cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)]))

    print()
    print('run again with the same input, to test caching..')
    print('schedules:', cohda.execute(target_schedule=target_schedule,
                        flexibility=[flex for i in range(n_agents)]))

if __name__ == '__main__':
    test()

