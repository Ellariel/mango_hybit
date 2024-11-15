import pandas as pd
import numpy as np
import pyswarms as ps
from itertools import chain
import random
import copy
import os

from ..utils import make_hashable
 
class SWARM():
    def __init__(self, n_particles=1000, n_iterations=1000, 
                 options={'c1': 1.4, 'c2': 1.4, 'w': 0.7},
                 verbose=False, **kwargs):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.options = options
        self.cache = {}

    def execute(self, target_schedule, flexibility, **kwargs):
            seed = kwargs.get('seed', None)
            cache_key = make_hashable((target_schedule, flexibility, seed))
            if cache_key in self.cache:
                if self.verbose:
                    print('SWARM returns a cached solution.')
                return copy.deepcopy(self.cache[cache_key])

            random.seed(seed)
            np.random.seed(seed)
            n_units = len(flexibility)
            n_points = len(target_schedule) 
            dimensions = n_points * n_units

            def cost_function(schedule, target=None):
                schedule = schedule.reshape(self.n_particles, n_units, n_points).sum(1)
                return np.sqrt(((np.asarray(target) - schedule)**2).sum(1))
            
            bounds = np.asarray([np.fromiter(chain.from_iterable([flex['flex_min_power'] for flex in flexibility]), float),
                                 np.fromiter(chain.from_iterable([flex['flex_max_power'] for flex in flexibility]), float)])
            init_pos = np.asarray([bounds[0]] * self.n_particles)

            optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, 
                                                dimensions=dimensions, 
                                                options=self.options, 
                                                bounds=bounds,
                                                init_pos=init_pos)

            cost, schedules = optimizer.optimize(objective_func=cost_function, 
                                        verbose=self.verbose, iters=self.n_iterations, 
                                        target=target_schedule)
            
            schedules = schedules.reshape(n_units, n_points)
            
            self.cache[cache_key] = schedules
            del optimizer

            return copy.deepcopy(self.cache[cache_key])

