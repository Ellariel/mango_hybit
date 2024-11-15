import pandas as pd
import numpy as np
import pyswarms as ps
import copy
import os

from mosaik_components.mas.utils import make_hashable
 
class SWARM():
    def __init__(self, n_particles=50, n_iterations=100, 
                 options={'c1': 1.4, 'c2': 1.4, 'w': 0.7},
                 verbose=False, **kwargs):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.options = options
        self.cache = {}

    def execute(self, target_schedule, flexibility, seed=None):
            cache_key = make_hashable((target_schedule, flexibility))
            if cache_key in self.cache:
                if self.verbose:
                    print('SWARM returns a cached solution.')
                return copy.deepcopy(self.cache[cache_key])

            np.random.seed(seed)
            dimensions = len(target_schedule)
            bounds = np.asarray((flexibility['flex_min_power'], flexibility['flex_max_power']))
            init_pos = np.asarray([flexibility['flex_min_power']] * self.n_particles)

            def cost_function(schedule, target=np.zeros(dimensions)):
                return np.sqrt(((np.asarray(target) - schedule)**2).sum(1))

            optimizer = ps.single.GlobalBestPSO(n_particles=self.n_particles, 
                                                dimensions=dimensions, 
                                                options=self.options, 
                                                bounds=bounds,
                                                init_pos=init_pos)

            cost, schedules = optimizer.optimize(objective_func=cost_function, 
                                        verbose=self.verbose, iters=self.n_iterations, 
                                        target=target_schedule)
            
            self.cache[cache_key] = schedules
            del optimizer

            return copy.deepcopy(self.cache[cache_key])

