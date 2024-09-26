import pandas as pd
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', default=1000, type=int)
parser.add_argument('--f', default=1, type=int)
parser.add_argument('--t', default=15, type=int)
args = parser.parse_args()

file_name = 'syntetic_loads_15min.csv'

random.seed(13)
data = pd.read_csv(file_name, skiprows=0)
data['Time'] = pd.to_datetime(data['Time'], format='mixed')
loads = [pd.Series([random.randint(args.f, args.t)*1.3 for j in range(len(data['Time']))]).rename(f'FLSim-{i}') for i in range(args.n)]
data = pd.concat([data['Time']] + loads, axis=1)
data.to_csv(file_name, index=False)

#with open(file_name, 'r') as original:
#    data = original.read()
#with open(file_name, 'w') as modified:
#    modified.write("Loads\n" + data)

