import numpy as np
import json
import logging
import random
import os


def set_device(gpu=-1):
    import torch
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def pprint(d, indent=0):
    d = sorted([(k, v) for k, v in d.items()], key=lambda x: x[0])
    for key, value in d:
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(round(value, 4)))


def seed_everything(seed=1029):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def print_to_json(data):
    new_data = dict((k, str(v)) for k, v in data.items())
    return json.dumps(new_data, indent=4, sort_keys=True)


def update_from_nni_params(params, nni_params):
    if nni_params:
        params.update(nni_params)
    return params
