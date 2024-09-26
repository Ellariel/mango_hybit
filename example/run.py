import os
for i, j in [('default', 'default'), 
             ('cohda', 'cohda'), 
             ('default', 'cohda'), 
             ('cohda', 'default')]:
    os.system(f"python scenario.py --within {i} --between {j}")