import argparse
import datetime
import pathlib
from typing import Any, Optional, Union

import termcolor
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import (CSVLogger, LightningLoggerBase,
                                       TensorBoardLogger, WandbLogger)
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only


def get_loggers(
        config: DictConfig,
        exp_root_dir: Union[str, pathlib.Path]) -> list[LightningLoggerBase]:
    loggers = []
    if config.LOG.USE_CSV_LOGGER:
        time_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        csv_logger = CSVLogger(
            save_dir=config.EXPERIMENT.ROOT_DIR,
            version=f'{config.EXPERIMENT.OUTPUT_DIR}/{time_str}',
            name=None)
        loggers.append(csv_logger)
    if config.LOG.USE_TENSORBOARD:
        tb_logger = TensorBoardLogger(
            save_dir=exp_root_dir,
            name=config.LOG.TENSORBOARD.LOG_DIR,
            version=config.EXPERIMENT.OUTPUT_DIR,
            default_hp_metric=config.LOG.TENSORBOARD.DEFAULT_HP_METRIC,
            purge_step=torch.load(config.EXPERIMENT.RESUME,
                                  map_location='cpu')['global_step']
            if config.EXPERIMENT.RESUME else None,
        )
        loggers.append(tb_logger)
    if config.LOG.USE_WANDB:
        run_id = None
        group = config.LOG.WANDB.GROUP
        if config.LOG.WANDB.RUN_ID:
            run_id = config.LOG.WANDB.RUN_ID
        elif config.EXPERIMENT.OUTPUT_DIR:
            generated_id = wandb.util.generate_id()
            if '/' in config.EXPERIMENT.OUTPUT_DIR:
                group_dir, group_id = config.EXPERIMENT.OUTPUT_DIR.split('/')
                run_id = f'{group_dir}-{group_id}-{generated_id}'
                group = group_dir
            else:
                run_id = f'{config.EXPERIMENT.OUTPUT_DIR}-{generated_id}'
        wandb_logger = WandbLogger(
            project=config.LOG.WANDB.PROJECT,
            save_dir=exp_root_dir,
            name=config.EXPERIMENT.OUTPUT_DIR,
            id=run_id,
            group=group,
        )
        loggers.append(wandb_logger)
    if config.LOG.USE_CONSOLE_LOGGER:
        loggers.append(ConsoleLogger(config))
    return loggers


class ExperimentWriter:
    def __init__(self, config: DictConfig):
        self.config = config

    def log_hparams(self, params: dict[str, Any]) -> None:
        if self.config.LOG.CONSOLE.SHOW_CONFIG and params:
            print(OmegaConf.to_yaml(DictConfig(params)))

    def _preprocess_dict(
        self, metrics_dict: dict[str, float]
    ) -> tuple[bool, bool, dict[str, float], dict[str, float]]:
        time_dict = dict()
        new_metrics_dict = dict()
        is_epoch = False
        is_train = False
        for key, val in metrics_dict.items():
            if '_epoch' in key:
                is_epoch = True
            if key.startswith('train/'):
                is_train = True
            if key.endswith('_epoch'):
                key = key[:-6]
            if key.endswith('_step'):
                key = key[:-5]
            if key.startswith('time/') or key == 'epoch':
                key = key.replace('time/', '')
                time_dict[key] = val
            else:
                key = key.replace('train/', '')
                key = key.replace('val/', '')
                new_metrics_dict[key] = val
        return is_epoch, is_train, new_metrics_dict, time_dict

    @staticmethod
    def _to_str(key: str,
                val: Union[float, int, str],
                color: str,
                total: Optional[Union[float, int]] = None) -> str:
        key = termcolor.colored(f'{key}:', color)
        n = len(str(total)) if total is not None else 0
        if isinstance(val, (str, int)):
            res = f'{key} {val:>{n}}'
        else:
            res = f'{key} {val: .4f}'
        if total:
            res += f'/{total}'
        return res

    def _create_time_label(self, time_dict: dict[str, Union[float, int]],
                           is_train: bool) -> str:
        label_color = self.config.LOG.CONSOLE.TRAIN_KEY_COLOR if is_train else self.config.LOG.CONSOLE.VAL_KEY_COLOR
        res = [
            self._to_str('epoch', time_dict['epoch'], label_color,
                         self.config.SCHEDULER.EPOCHS),
        ]
        if 'train_batch_index' in time_dict:
            res.append(
                self._to_str('step', time_dict['train_batch_index'],
                             label_color, time_dict['train_total_batches']))

        for key, val in time_dict.items():
            if key in ['train_speed', 'val_speed']:
                res.append(
                    self._to_str('speed', f'{val:.2f} it/s', label_color))
            else:
                time_str = str(datetime.timedelta(seconds=round(val)))
                if key in [
                        'eta',
                        'epoch_eta',
                        'train_epoch_time',
                        'val_epoch_time',
                        'elapsed',
                ]:
                    res.append(self._to_str(key, time_str, label_color))
        return ', '.join(res)

    def _create_metrics_label(self, metrics_dict: dict[str, float],
                              is_epoch: bool, is_train: bool) -> str:
        label_color = self.config.LOG.CONSOLE.TRAIN_KEY_COLOR if is_train else self.config.LOG.CONSOLE.VAL_KEY_COLOR
        res = []
        for key, val in metrics_dict.items():
            res.append(self._to_str(key, val, label_color))
        epoch_or_step = 'epoch' if is_epoch else 'step'
        stage_label = f'[train_{epoch_or_step}] ' if is_train else '[val] '
        stage_label = termcolor.colored(
            stage_label, self.config.LOG.CONSOLE.STAGE_NAME_COLOR)
        return stage_label + ', '.join(res)

    def log_metrics(self,
                    metrics_dict: dict[str, float],
                    step: Optional[int] = None) -> None:
        is_epoch, is_train, metrics_dict, time_dict = self._preprocess_dict(
            metrics_dict)

        res = [
            termcolor.colored(f'[{datetime.datetime.now()}]',
                              self.config.LOG.CONSOLE.TIMESTAMP_COLOR),
            self._create_time_label(time_dict, is_train),
            self._create_metrics_label(metrics_dict, is_epoch, is_train),
        ]
        print(' '.join(res))


class ConsoleLogger(LightningLoggerBase):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self._experiment = None

    @property  # type: ignore
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        if self._experiment:
            return self._experiment
        self._experiment = ExperimentWriter(self.config)  # type: ignore
        return self._experiment  # type: ignore

    @rank_zero_only
    def log_hyperparams(
            self, params: Union[dict[str, Any], argparse.Namespace]) -> None:
        params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self,
                    metrics: dict[str, float],
                    step: Optional[int] = None) -> None:
        self.experiment.log_metrics(metrics, step)

    @property
    def name(self):
        return 'ConsoleLogger'

    @property
    def version(self):
        return '0.0'
