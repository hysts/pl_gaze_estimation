import functools
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig


def convert_to_unit_vector(angles: torch.Tensor) -> torch.Tensor:
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = torch.cos(pitches) * torch.sin(yaws)
    y = torch.sin(pitches)
    z = torch.cos(pitches) * torch.cos(yaws)
    vector = torch.cat([x[:, None], y[:, None], z[:, None]], dim=1)
    norm = torch.norm(vector, dim=1, keepdim=True)
    return vector / norm


def compute_angle_error(predictions: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    pred = convert_to_unit_vector(predictions)
    gt = convert_to_unit_vector(labels)
    inner_product = (pred * gt).sum(dim=1)
    inner_product = torch.clip(inner_product, min=-1, max=1)
    return torch.acos(inner_product) * 180 / np.pi


def get_loss_func(config: DictConfig) -> Callable:
    loss_name = config.MODEL.LOSS.TYPE
    if loss_name == 'l1':
        return functools.partial(F.l1_loss, reduction='mean')
    elif loss_name == 'l2':
        return functools.partial(F.mse_loss, reduction='mean')
    elif loss_name == 'smooth_l1':
        return functools.partial(F.smooth_l1_loss, reduction='mean')
    else:
        raise ValueError
