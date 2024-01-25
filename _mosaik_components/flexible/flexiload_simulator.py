from __future__ import annotations

import arrow
import pandas as pd
from os.path import abspath
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import mosaik_api_v3
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
        "FLSim": {
            "public": True,
            "any_inputs": True,
            "persistent": [],
            "params": ["scale_factor"], 
            "attrs": ["P[MW]",           # input/output active power [MW]
                      'scale_factor'],   # input of modifier from ctrl
        }
    },
}

STEP_SIZE = 60*60 
CACHE_DIR = Path(abspath(__file__)).parent
DATE_FORMAT = "YYYY-MM-DD HH:mm:ss"

class FLSimulator(mosaik_api_v3.Simulator):
    _sid: str
    """This simulator's ID."""
    _step_size: Optional[int]
    """The step size for this simulator. If ``None``, the simulator
    is running in event-based mode, instead.
    """
    sim_params: Dict
    """Simulator parameters specification:
    SIM_PARAMS = {
        'start_date' : '2016-01-01 00:00:00',
        'gen_neg' : True,
    } 
    """

    def __init__(self) -> None:
        super().__init__(META)
    
    def init(self, sid: str, time_resolution: float = 1, step_size: int = STEP_SIZE, sim_params: Dict = {}):
        self.gen_neg = sim_params.get('gen_neg', False)
        self.date = arrow.get(sim_params.get('start_date', '2016-01-01 00:00:00'), DATE_FORMAT)
        self.time_resolution = time_resolution
        self.step_size = step_size
        self._first_step = True
        self.sid = sid
        self.entities = {}
        self.scale_factor = {}
        return self.meta

    def create(self, num: int, model: str, **model_params: Any) -> List[CreateResult]:
        entities = []
        for n in range(len(self.entities), len(self.entities) + num):
            eid = f"{model}-{n}"
            self.entities[eid] = 0
            self.scale_factor[eid] = 1 # Default value
            entities.append({
                "eid": eid,
                "type": model,
            })
        return entities
    
    def _get_data(self, eid, attr):
        if attr == 'scale_factor':
            return self.scale_factor[eid]
        result = self.entities[eid] * self.scale_factor[eid]
        if self.gen_neg:
            result *= (-1)
        return result

    def step(self, time, inputs, max_advance):
        if not self._first_step:
            self.date = self.date.shift(seconds=self.step_size)
        self._first_step = False
        print('FL!!!!!!!', inputs)
        for eid, attrs in inputs.items():
            for attr, vals in attrs.items():
                if attr == 'P[MW]':
                    self.entities[eid] = list(vals.values())[0]
                elif attr == 'scale_factor':
                    self.scale_factor[eid] = list(vals.values())[0]
                else:
                    pass

        return time + self.step_size
     
    def get_data(self, outputs: OutputRequest) -> OutputData:
        return {eid: {attr: self._get_data(eid, attr) 
                            for attr in attrs
                                } for eid, attrs in outputs.items()}