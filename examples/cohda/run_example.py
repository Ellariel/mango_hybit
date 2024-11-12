import os
os.system(f"python grid_model.py")
os.system(f"python grid_profiles.py")
for i, j in [('default', 'default'), 
             ('cohda', 'cohda'), 
             ('default', 'cohda'), 
             ('cohda', 'default')]:
    print(f"python scenario.py --within {i} --between {j}")
    os.system(f"python scenario.py --within {i} --between {j}")
os.remove('mas.log')

# outputs are in the results folder 