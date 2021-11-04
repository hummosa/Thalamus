import os
import importlib
from models import MODES


all_configs = {}
module = importlib.import_module('configs.configs')
for mode in MODES:
    all_configs[mode] = getattr(module, mode+'Config')

def get_config(mode):
    assert mode in MODES, "Please choose a mode from "+str(MODES)
    config = all_configs[mode]()
    config.mode = mode
    return config