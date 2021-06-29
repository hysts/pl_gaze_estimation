from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .backbones import create_backbone


class Network(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.feature_extractor = create_backbone(config)
        n_features = self.feature_extractor.n_features
        feature_map_size = self.feature_extractor.feature_map_size

        self.conv = nn.Conv2d(n_features,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.fc = nn.Linear(n_features * feature_map_size**2, 2)

        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _register_hook(self):
        n_features = self.feature_extractor.n_features

        def hook(
            module: nn.Module, grad_in: Union[Tuple[torch.Tensor, ...],
                                              torch.Tensor],
            grad_out: Union[Tuple[torch.Tensor, ...], torch.Tensor]
        ) -> Optional[torch.Tensor]:
            return tuple(grad / n_features for grad in grad_in)

        self.conv.register_full_backward_hook(hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        y = F.relu(self.conv(x))
        x = x * y
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
