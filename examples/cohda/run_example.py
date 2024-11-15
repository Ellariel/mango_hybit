import os
import itertools

os.system(f"python grid_model.py")
os.system(f"python grid_profiles.py")
alg = ['default', 'cohda', 'swarm']
for i, j in itertools.product(alg, repeat=2):
    print(f"python scenario.py --within {i} --between {j}")
    os.system(f"python scenario.py --within {i} --between {j}")

# outputs are in the results folder 