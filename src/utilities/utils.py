import json
import os

def load_config(config_name):

    config_dir = os.path.join(os.path.dirname(__file__),'..','..','configs')
    file_path = os.path.join(config_dir, config_name)

    with open(file_path, 'r') as file:
        config = json.load(file)

    return config