import os
os.system(f"python grid_model.py")
os.system(f"python grid_profiles.py")
for i, j in [('swarm', 'swarm'), 
             ('cohda', 'cohda'), 
             ('swarm', 'cohda'), 
             ('cohda', 'swarm')]:
    print(f"python scenario.py --within {i} --between {j}")
    os.system(f"python scenario.py --within {i} --between {j}")

# outputs are in the results folder 