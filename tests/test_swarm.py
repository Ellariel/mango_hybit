import time
import numpy as np
from massca.lib.swarm import SWARM


def test_swarm():
    """
    Run test.
    """

    swarm = SWARM(verbose=True)
    target_schedule = [1.5, 20.0, 15.0]
    flexibility = {'flex_max_power': [3.0, 5.0, 7.0],
                'flex_min_power': [0.0, 0.1, 0.3],
        }
    n_units = 5
    print('n_units:', n_units)
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = swarm.execute(target_schedule=target_schedule,
                                flexibility=[flexibility for i in range(n_units)], seed=42)
    print('schedules:', schedules)
    print('solution:', schedules.sum(0))
    rmse = np.sqrt(np.sum((target_schedule - schedules.sum(0))**2)/len(target_schedule))
    print('RMSE:', rmse)
    assert len(schedules) == n_units
    assert len(schedules[0]) == len(target_schedule)
    assert rmse < 0.08

    time.sleep(1)
    print()

    target_schedule = [0.1, 0.0, 2.0]
    flexibility = {'flex_max_power': [0.1, 1.0, 1.0],
            'flex_min_power': [0.0, 0.0, 0.0],
            }
    n_units = 3
    print('n_units:', n_units)
    print('target_schedule:', target_schedule)
    print('flexibility:', flexibility)
    schedules = swarm.execute(target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_units)], seed=13)
    print('schedules:', schedules)
    print('solution:', schedules.sum(0))
    rmse = np.sqrt(np.sum((target_schedule - schedules.sum(0))**2)/len(target_schedule))
    print('RMSE:', rmse)
    assert rmse < 0.03
    last_schedules = schedules

    print()
    print('run again with the same input, to test caching..')
    schedules = swarm.execute(target_schedule=target_schedule,
                        flexibility=[flexibility for i in range(n_units)], seed=13)
    print('schedules:', schedules)
    assert (schedules == last_schedules).all()

if __name__ == '__main__':
    test_swarm()

