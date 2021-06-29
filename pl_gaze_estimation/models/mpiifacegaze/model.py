import importlib
import logging
import time

import torch
from omegaconf import DictConfig

from ...pl_utils.model import Model as PlModel
from ..utils.gaze import compute_angle_error, get_loss_func
from ..utils.optimizer import configure_optimizers
from ..utils.utils import initialize_weight, load_weight

logger = logging.getLogger(__name__)


def create_torch_model(config: DictConfig) -> torch.nn.Module:
    module = importlib.import_module(
        f'pl_gaze_estimation.models.mpiifacegaze.models.{config.MODEL.NAME}')
    model = getattr(module, 'Network')(config)
    if 'INIT' in config.MODEL:
        initialize_weight(config.MODEL.INIT, model)
    else:
        logger.warning('INIT key is missing in config.MODEL.')
    if 'PRETRAINED' in config.MODEL:
        load_weight(config.MODEL, model)
    else:
        logger.warning('PRETRAINED key is missing in config.MODEL.')
    return model


class Model(PlModel):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.model = create_torch_model(config)
        if 'OPTIMIZER' in self.config:
            self.lr = self.config.OPTIMIZER.LR
        self.loss_fn = get_loss_func(config)

    def forward(self, x):
        return self.model(x)

    def _evaluate(self, batch):
        images, _, gazes = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, gazes)
        angle_error = compute_angle_error(outputs, gazes).mean()
        return {'loss': loss, 'angle_error': angle_error}

    def training_step(self, batch, batch_index):
        res = self._evaluate(batch)
        self.log_dict({f'train/{key}': val
                       for key, val in res.items()},
                      prog_bar=self.config.LOG.SHOW_TRAIN_IN_PROGRESS_BAR,
                      on_step=True,
                      on_epoch=True,
                      logger=True,
                      sync_dist=True,
                      sync_dist_op='mean')
        self.log_dict(
            {
                'time/elapsed':
                time.time() - self._tic + self._train_time + self._val_time
            },
            on_step=True,
            on_epoch=False,
            logger=True)
        if self.lr_schedulers():
            if isinstance(self.lr_schedulers(), list):
                scheduler = self.lr_schedulers()[-1]
            else:
                scheduler = self.lr_schedulers()
            self.log('train/lr',
                     scheduler.get_last_lr()[0],
                     prog_bar=self.config.LOG.SHOW_TRAIN_IN_PROGRESS_BAR,
                     on_step=True,
                     on_epoch=False,
                     logger=True)

        return res

    def validation_step(self, batch, batch_index):
        res = self._evaluate(batch)
        return res | {
            'size': torch.tensor(len(batch[0]), device=res['loss'].device)
        }

    def validation_epoch_end(self, outputs) -> None:
        res = self._accumulate_data(outputs)
        self.log_dict(
            {f'val/{key}': val
             for key, val in res.items() if key != 'total'},
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=False)

    def test_step(self, batch, batch_index):
        return self.validation_step(batch, batch_index)

    def test_epoch_end(self, outputs) -> None:
        res = self._accumulate_data(outputs)
        self._display_result(res)

    def configure_optimizers(self):
        return configure_optimizers(self.config, self, self.lr)
