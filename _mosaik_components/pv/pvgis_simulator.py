from __future__ import annotations

import arrow
import pandas as pd
from os.path import abspath
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import mosaik_api_v3
from mosaik_components.pv.pvgis import PVGIS
from mosaik_api_v3.types import (
    CreateResult,
    CreateResultChild,
    Meta,
    ModelDescription,
    OutputData,
    OutputRequest,
)

META = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "PVSim": {
            "public": True,
            "any_inputs": True,
            "persistent": [],
            "params": ["scale_factor", "slope", "azimuth", "pvtech", 
                        "lat", "lon", "system_loss", "datayear", "datatype",
                        "optimal_angle", "optimal_both", "database"], 
            "attrs": ["P[MW]",           # estimated active PV power supply based on reference year [MW]
                      'scale_factor'],   # input of modifier from ctrl
        }
    },
}

STEP_SIZE = 60*60 
CACHE_DIR = Path(abspath(__file__)).parent
DATE_FORMAT = "YYYY-MM-DD HH:mm:ss"

class PVGISSimulator(mosaik_api_v3.Simulator):
    _sid: str
    """This simulator's ID."""
    _step_size: Optional[int]
    """The step size for this simulator. If ``None``, the simulator
    is running in event-based mode, instead.
    """
    sim_params: Dict
    """Simulator parameters specification:
    PVSIM_PARAMS = {
        'start_date' : '2016-01-01 00:00:00',
        'cache_dir' : './',
        'verbose' : True,
    } 
    """

    def __init__(self) -> None:
        super().__init__(META)
    
    def init(self, sid: str, time_resolution: float = 1, step_size: int = STEP_SIZE, sim_params: Dict = {}):
        self.cache_dir = sim_params.get('cache_dir', str(CACHE_DIR))
        self.verbose = sim_params.get('verbose', True)
        self.gen_neg = sim_params.get('gen_neg', False)
        self.date = arrow.get(sim_params.get('start_date', '2016-01-01 00:00:00'), DATE_FORMAT)
        self.time_resolution = time_resolution
        self.step_size = step_size
        self._first_step = True
        self.sid = sid
        self.pvgis = PVGIS(verbose=self.verbose, 
                           local_cache_dir=self.cache_dir)
        self.entities = {}
        self.scale_factor = {}
        return self.meta

    def create(self, num: int, model: str, **model_params: Any) -> List[CreateResult]:
        entities = []
        for n in range(len(self.entities), len(self.entities) + num):
            eid = f"{model}-{n}"
            production, info = self.pvgis.get_production_timeserie(**model_params)
            if self.verbose:
                print('model_params:', model_params, 'info:', info)
            production.index = pd.to_datetime(production.index, utc=True) +\
                        pd.offsets.DateOffset(years=self.date.year - production.index[0].year) # change history year to current one

            old_index = production.index.copy()
            new_step_size = pd.Timedelta(self.step_size * self.time_resolution, unit='seconds')
            production = production.resample(new_step_size).sum()
            new_index = production.index.get_indexer(old_index, method='ffill')

            for i in range(0, len(new_index) - 1): # rescaling with new step size
                production.iloc[new_index[i]:new_index[i+1]] = production.iloc[new_index[i]:new_index[i+1]].mean()

            self.entities[eid] = production / 10**6 # W per 1 kW peak -> MW
            self.scale_factor[eid] = 1 # Default value
            entities.append({
                "eid": eid,
                "type": model,
            })
        return entities
    
    def get_production(self, eid, attr):
        idx = self.entities[eid].index.get_indexer([self.date.datetime], method='ffill')[0]
        production = self.entities[eid].iloc[idx] * self.scale_factor[eid]
        if self.gen_neg:
            production *= (-1)
        return production

    def step(self, time, inputs, max_advance):
        if not self._first_step:
            self.date = self.date.shift(seconds=self.step_size)
        self._first_step = False
        print('!!!!!!!', inputs)
        for eid, attrs in inputs.items():
            for attr, vals in attrs.items():
                if attr == 'scale_factor':
                    self.scale_factor[eid] = list(vals.values())[0]

        return time + self.step_size
     
    def get_data(self, outputs: OutputRequest) -> OutputData:
        data =  {eid: {attr: self.get_production(eid, attr) 
                            for attr in attrs
                                } for eid, attrs in outputs.items()}
        #print(data)
        return data