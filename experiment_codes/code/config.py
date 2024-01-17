import os

# getting the parent directory and changing the current working directory to the parent directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
parent_dname = os.path.dirname(dname)

os.chdir(parent_dname)

ROOT_DIR = parent_dname