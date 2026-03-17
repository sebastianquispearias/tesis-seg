import json
import os
import random
from copy import deepcopy

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_parameters_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def clone_model_weights(student, teacher):
    teacher.load_state_dict(deepcopy(student.state_dict()))


@torch.no_grad()
def update_ema(student, teacher, ema_decay: float = 0.99):
    sdict = student.state_dict()
    tdict = teacher.state_dict()

    for k in tdict.keys():
        if tdict[k].dtype.is_floating_point:
            tdict[k].mul_(ema_decay).add_(sdict[k], alpha=1.0 - ema_decay)
        else:
            tdict[k].copy_(sdict[k])