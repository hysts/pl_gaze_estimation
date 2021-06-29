import importlib

import torch
from omegaconf import DictConfig


def create_model(config: DictConfig) -> torch.nn.Module:
    module = importlib.import_module(
        f'pl_gaze_estimation.models.{config.MODEL.TYPE}.model')
    model = getattr(module, 'Model')(config)
    return model
