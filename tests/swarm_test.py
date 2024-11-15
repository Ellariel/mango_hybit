import time
import numpy as np
from swarm import SWARM


def test_swarm():
    """
    Run test.
    """

    swarm = SWARM(verbose=True)
    target_schedule = [0.5, 20.0, 5.0]
    flexibility = {'flex_max_power': [3.0, 30.0, 7.0],
                'flex_min_power': [0.1, 5.0, 0.3],
        }

    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = swarm.execute(target_schedule=target_schedule,
                                flexibility=flexibility, seed=42)
    print('schedules:', schedules)

    time.sleep(1)
    print()

    target_schedule = [0.1, 0.0, 2.0]
    flexibility = {'flex_max_power': [0.1, 1.0, 1.0],
            'flex_min_power': [0.0, 0.0, 0.0],
            }
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = swarm.execute(target_schedule=target_schedule,
                        flexibility=flexibility, seed=13)
    print('schedules:', schedules)
    last_schedules = schedules

    print()
    print('run again with the same input, to test caching..')
    schedules = swarm.execute(target_schedule=target_schedule,
                        flexibility=flexibility, seed=1313)
    print('schedules:', schedules)

if __name__ == '__main__':
    test_swarm()

