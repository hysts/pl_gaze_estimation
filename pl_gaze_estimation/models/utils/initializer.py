from typing import Callable

import torch.nn as nn
from omegaconf import DictConfig


def create_initializer(init_config: DictConfig) -> Callable:
    if 'CONV' in init_config and 'TYPE' in init_config.CONV:
        assert init_config.CONV.TYPE in [
            'kaiming_normal',
            'kaiming_uniform',
            'xavier_normal',
            'xavier_uniform',
        ]
    if 'LINEAR' in init_config and 'TYPE' in init_config.LINEAR:
        assert init_config.LINEAR.TYPE in [
            'kaiming_normal',
            'kaiming_uniform',
            'xavier_normal',
            'xavier_uniform',
        ]

    def initializer(module):
        if isinstance(module, nn.Conv2d) and 'CONV' in init_config:
            if 'TYPE' in init_config.CONV:
                if init_config.CONV.TYPE == 'kaiming_normal':
                    nn.init.kaiming_normal_(
                        module.weight.data,
                        mode=init_config.CONV.KAIMING.MODE,
                        nonlinearity=init_config.CONV.KAIMING.NONLINEARITY)
                elif init_config.CONV.TYPE == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(
                        module.weight.data,
                        mode=init_config.CONV.KAIMING.MODE,
                        nonlinearity=init_config.CONV.KAIMING.NONLINEARITY)
                elif init_config.CONV.TYPE == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight.data,
                                           gain=init_config.CONV.XAVIER.GAIN)
                elif init_config.CONV.TYPE == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight.data,
                                            gain=init_config.CONV.XAVIER.GAIN)
            if (module.bias is not None and 'BIAS' in init_config.CONV
                    and init_config.CONV.BIAS == 0):
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Linear) and 'LINEAR' in init_config:
            if 'TYPE' in init_config.LINEAR:
                if init_config.LINEAR.TYPE == 'kaiming_normal':
                    nn.init.kaiming_normal_(
                        module.weight.data,
                        mode=init_config.LINEAR.KAIMING.MODE,
                        nonlinearity=init_config.LINEAR.KAIMING.NONLINEARITY)
                elif init_config.LINEAR.TYPE == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(
                        module.weight.data,
                        mode=init_config.LINEAR.KAIMING.MODE,
                        nonlinearity=init_config.LINEAR.KAIMING.NONLINEARITY)
                elif init_config.LINEAR.TYPE == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight.data,
                                           gain=init_config.LINEAR.XAVIER.GAIN)
                elif init_config.LINEAR.TYPE == 'xavier_uniform':
                    nn.init.xavier_uniform_(
                        module.weight.data,
                        gain=init_config.LINEAR.XAVIER.GAIN)
            if (module.bias is not None and 'BIAS' in init_config.LINEAR
                    and init_config.LINEAR.BIAS == 0):
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.BatchNorm2d) and 'BN' in init_config:
            if init_config.BN.WEIGHT == 1:
                nn.init.ones_(module.weight.data)
            if init_config.BN.BIAS == 0:
                nn.init.zeros_(module.bias.data)

    return initializer
