import os
import sys
import time
import json
from pathlib import Path
import argparse
import pandapower as pp
import pandas as pd
import numpy as np
import random
import arrow
from pandas.io.json._normalize import nested_to_record

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def timeseries(start='2016-01-01 00:00:00', end=60*60, step_size=60*15):
    date_start = arrow.get(start, 'YYYY-MM-DD hh:mm:ss')
    return pd.Series([i.format('YYYY-MM-DD HH:mm:ss') 
                for i in list(arrow.Arrow.range('seconds', date_start, 
                        date_start.shift(seconds=end)))[::int(step_size)]]).rename('Time')

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./', type=str)
parser.add_argument('--start', default='2016-01-01 00:00:00', type=str)
parser.add_argument('--end', default=60*60, type=int)
parser.add_argument('--step', default=60*15, type=int)
parser.add_argument('--seed', default=13, type=int)
args = parser.parse_args()

base_dir = args.dir
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
grid_file = os.path.join(data_dir, 'grid_model.json')
prof_file = os.path.join(data_dir, 'grid_profiles.csv')
grid = pp.from_json(grid_file)
pp.runpp(grid, numba=False)
print(f"Grid model of {len(grid.load)} loads, {len(grid.sgen)} sgens, {len(grid.bus)} buses, {len(grid.line)} lines, {len(grid.trafo)} trafos")

random.seed(args.seed)
np.random.seed(args.seed)

timeline = timeseries(start=args.start, end=args.end, step_size=args.step)
units = {}
loads = {f"Load-{idx}" : {'min' : min(1, i['p_mw']),
                        'max' : max(5, i['p_mw'] * 5),
                        'value' : i['p_mw'],
                        'delta' : 0,
                        } for idx, i in grid.load.iterrows()}
sgens = {f"StaticGen-{idx}" : {'min' : min(1, i['p_mw']),
                        'max' : max(5, i['p_mw'] * 5),
                        'value' : i['p_mw'],
                        'delta' : 0,
                        } for idx, i in grid.sgen.iterrows()}
units.update(loads)
units.update(sgens)
profiles = pd.DataFrame()
for k, v in nested_to_record(units, sep='.').items():
        if "min" in k:
                a, b = v, v + 0.1
        elif "max" in k:
                a, b = v - 0.5, v + 0.5
        elif "value" in k:
                a, b = v - 0.5, v + 0.5
        else:
                a, b = 0, 0
        profiles = pd.concat([profiles,
                              pd.Series([random.uniform(a, b) 
                                                for i in range(len(timeline))]).rename(k)], axis=1)

for k, v in units.items():
        profiles[f"{k}.value"] += profiles[f"{k}.min"]

profiles = pd.concat([timeline, profiles], axis=1)

profiles.to_csv(prof_file, index=False)
profiles = Path(prof_file)
profiles.write_text(f"Profiles\n{profiles.read_text()}")

print(f"Start: {timeline.iloc[0]}\nEnd: {timeline.iloc[-1]}\nStep: {args.step} sec")

print(f"Profiles were saved: {prof_file}")