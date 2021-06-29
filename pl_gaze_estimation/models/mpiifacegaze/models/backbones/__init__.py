import importlib

import torch.nn as nn
from omegaconf import DictConfig


def create_backbone(config: DictConfig) -> nn.Module:
    backbone_name = config.MODEL.BACKBONE.NAME
    module = importlib.import_module(
        f'pl_gaze_estimation.models.mpiifacegaze.models.backbones.{backbone_name}'
    )
    return module.Network(config)  # type: ignore
