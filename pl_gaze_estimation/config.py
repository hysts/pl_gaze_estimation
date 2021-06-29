import argparse
import logging
import pathlib
from typing import Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs',
                        type=str,
                        nargs='*',
                        required=True,
                        help='Paths to config files.')
    parser.add_argument('--options',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='Variables to overwrite. (optional)')
    args = parser.parse_args()

    config = load_configs(args.configs)
    cli_config = options_to_config(args.options, without_equal_sign=True)
    check_if_cli_keys_in_config(config, cli_config)
    config = OmegaConf.merge(config, cli_config)
    if 'DEBUG' in config and config.DEBUG.DEBUG:
        config.LOG.USE_TENSORBOARD = False
        config.LOG.USE_WANDB = False

    OmegaConf.set_readonly(config, True)
    return config


def load_configs(paths: list[Union[str, pathlib.Path]]) -> DictConfig:
    return OmegaConf.merge(*[OmegaConf.load(path) for path in paths])


def options_to_config(options: list[str],
                      without_equal_sign: bool) -> DictConfig:
    if not options:
        return OmegaConf.create()
    if without_equal_sign:
        keys = options[::2]
        vals = options[1::2]
        options = list(map(lambda x: f'{x[0]}={x[1]}', zip(keys, vals)))
    return OmegaConf.from_cli(options)


def get_flatten_keys(config: dict) -> list[str]:
    res = []
    for key, val in config.items():
        if not isinstance(val, dict):
            res.append(key)
        else:
            res.extend([f'{key}.{name}' for name in get_flatten_keys(val)])
    return res


def check_if_cli_keys_in_config(config: DictConfig,
                                cli_config: DictConfig) -> None:
    base_keys = set(get_flatten_keys(OmegaConf.to_container(config)))
    cli_keys = set(get_flatten_keys(OmegaConf.to_container(cli_config)))
    unexpected_keys = sorted(cli_keys - base_keys)
    if unexpected_keys:
        logger.error(
            'It is not allowed to use --options to add new keys to config.')
        logger.error('The following keys are unexpected:')
        for key in unexpected_keys:
            logger.error(f'  {key}')
        raise RuntimeError
