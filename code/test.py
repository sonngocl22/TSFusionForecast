from config import ROOT_DIR

import os
import sys

if ROOT_DIR not in sys.path:
    # Add the parent directory to sys.path
    sys.path.insert(0, ROOT_DIR)

import utilities

config = utilities.load_config("test.json")

print(config)