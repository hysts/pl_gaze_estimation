from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_warmup_scheduler import WarmupScheduler


def get_param_list(config: DictConfig, model: nn.Module,
                   weight_decay: float) -> list[dict[str, Any]]:
    param_list = []
    for module in model.modules():
        if not hasattr(module, 'weight'):
            continue
        if config.OPTIMIZER.NO_WEIGHT_DECAY_ON_BN and isinstance(
                module, nn.BatchNorm2d):
            for params in module.parameters():
                param_list.append({
                    'params': params,
                    'weight_decay': 0,
                })
        elif config.OPTIMIZER.NO_WEIGHT_DECAY_ON_BIAS and isinstance(
                module, (nn.Conv2d, nn.Linear)):
            for name, params in module.named_parameters():
                if name == 'bias':
                    param_list.append({
                        'params': params,
                        'weight_decay': 0,
                    })
                else:
                    param_list.append({
                        'params': params,
                        'weight_decay': weight_decay,
                    })
        else:
            for params in module.parameters():
                param_list.append({
                    'params': params,
                    'weight_decay': weight_decay,
                })
    return param_list


def configure_optimizers(config: DictConfig, model: nn.Module, lr: float):
    params = get_param_list(config, model, config.OPTIMIZER.WEIGHT_DECAY)
    if config.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=lr,
                                    momentum=config.OPTIMIZER.SGD.MOMENTUM,
                                    nesterov=config.OPTIMIZER.SGD.NESTEROV)
    elif config.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=lr,
                                     betas=config.OPTIMIZER.ADAM.BETAS,
                                     amsgrad=config.OPTIMIZER.ADAM.AMSGRAD)
    elif config.OPTIMIZER.NAME == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=lr,
                                      betas=config.OPTIMIZER.ADAMW.BETAS,
                                      amsgrad=config.OPTIMIZER.ADAMW.AMSGRAD)
    else:
        raise ValueError

    if 'SCHEDULER' in config:
        if config.SCHEDULER.NAME == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, config.SCHEDULER.EPOCHS,
                config.SCHEDULER.COSINE.LAST_LR)
        elif config.SCHEDULER.NAME == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.SCHEDULER.MULTISTEP.MILESTONES,
                gamma=config.SCHEDULER.MULTISTEP.GAMMA)
        else:
            raise ValueError
        schedulers = [{
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }]
        # Because `torch.optim.lr_scheduler.CosineAnnealingScheduler`
        # keeps `optimizer.param_groups[0]['lr']` etc. as
        # `self.base_lrs` in `__init__()`, and the `WarmupScheduler`
        # changes them, so it needs to be instantiated after the base
        # scheduler is instantiated.
        if ('WARMUP' in config.SCHEDULER
                and config.SCHEDULER.WARMUP.EPOCHS > 0):
            warmup_scheduler = WarmupScheduler(
                optimizer,
                warmup_epoch=config.SCHEDULER.WARMUP.EPOCHS,
                initial_lr_factor=config.SCHEDULER.WARMUP.INITIAL_LR_FACTOR)
            schedulers.append({
                'scheduler': warmup_scheduler,
                'interval': 'epoch',
                'frequency': 1,
            })
        return [optimizer], schedulers
    else:
        return optimizer
