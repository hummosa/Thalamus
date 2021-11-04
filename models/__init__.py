# Ref of baseline:
# https://github.com/aimagelab/mammoth
# https://github.com/shivamsaboo17/Overcoming-Catastrophic-forgetting-in-Neural-Networks

import os
import importlib


MODES = ['Base', 'EWC', 'SI']
all_models = {}
module = importlib.import_module('models.baselines')
for mode in MODES:
    all_models[mode] = getattr(module, mode)

def get_model(backbone, loss, config, transform, opt, device, parameters, named_parameters):
    return all_models[config.mode](backbone, loss, config, transform, opt, device, parameters, named_parameters)