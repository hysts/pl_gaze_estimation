import torch
from omegaconf import DictConfig

from .initializer import create_initializer


def initialize_weight(init_config: DictConfig, model: torch.nn.Module) -> None:
    initializer = create_initializer(init_config)
    model.apply(initializer)


def load_weight(model_config: DictConfig, model: torch.nn.Module) -> None:
    if 'PRETRAINED' not in model_config or not model_config.PRETRAINED.PATH:
        return
    checkpoint = torch.load(model_config.PRETRAINED.PATH, map_location='cpu')
    state_dict = checkpoint[model_config.PRETRAINED.KEY]
    for key, val in list(state_dict.items()):
        remove_prefix = model_config.PRETRAINED.REMOVE_PREFIX
        add_prefix = model_config.PRETRAINED.ADD_PREFIX
        new_key = key
        if remove_prefix:
            new_key = new_key[len(remove_prefix) + 1:]
        if add_prefix:
            new_key = f'{add_prefix}.{new_key}'
        if new_key != key:
            state_dict[new_key] = val
            del state_dict[key]
    model.load_state_dict(state_dict)
