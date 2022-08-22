import yaml
import torch
import numpy as np

import random
import os
from typing import Dict, List

from datasets import HumanDataset
from collections import OrderedDict


def load_yaml(file_name: str) -> Dict:
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_pytorch_model(state_dict, *args, **kwargs):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('model.'):
            # print(name)
            name = name.replace('.model.', '.')
        new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    return new_state_dict
