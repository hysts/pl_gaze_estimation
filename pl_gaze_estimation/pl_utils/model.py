import time
from typing import Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.distributed import (ReduceOp, rank_zero_only,
                                                     sync_ddp_if_available)
from torch.optim import Optimizer


class Model(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self._train_time = 0.
        self._val_time = 0.
        self._tic = time.time()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self._train_time = checkpoint['train_time']
        self._val_time = checkpoint['val_time']

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['train_time'] = self._train_time
        checkpoint['val_time'] = self._val_time

    def on_train_start(self) -> None:
        if isinstance(self.logger, list):
            for logger in self.logger:
                logger.log_hyperparams(self.config)
        else:
            self.logger.log_hyperparams(self.config)

    def on_train_epoch_start(self) -> None:
        self._tic = time.time()

    def on_train_epoch_end(self, unused=None) -> None:
        elapsed = time.time() - self._tic
        self._train_time += elapsed
        self.log_dict(
            {
                'time/train_epoch_time': elapsed,
                'time/train_total_time': self._train_time,
            },
            on_epoch=True,
            logger=True)

    def on_validation_epoch_start(self) -> None:
        self._tic = time.time()

    def _accumulate_data(self, outputs):
        keys = [key for key in outputs[0].keys() if key != 'size']

        total = torch.zeros(1, dtype=torch.int, device=self.device)
        res = {
            key: torch.zeros(1,
                             dtype=outputs[0][key].dtype,
                             device=self.device)
            for key in keys
        }
        for batch in outputs:
            size = batch['size']
            total += size
            for key in keys:
                res[key] += batch[key] * size

        total = sync_ddp_if_available(total, reduce_op=ReduceOp.SUM)
        for key in keys:
            res[key] = sync_ddp_if_available(res[key], reduce_op=ReduceOp.SUM)
            res[key] /= total

        return res | {'total': total}

    def on_validation_epoch_end(self) -> None:
        elapsed = time.time() - self._tic
        self._val_time += elapsed
        self.log_dict(
            {
                'time/val_epoch_time': elapsed,
                'time/val_total_time': self._val_time,
            },
            on_epoch=True,
            logger=True)

    @rank_zero_only
    def _display_result(self, data) -> None:
        print({key: round(val.item(), 4) for key, val in data.items()})

    def test_epoch_end(self, outputs) -> None:
        res = self._accumulate_data(outputs)
        self._display_result(res)

    def optimizer_zero_grad(self, epoch: int, batch_idx: int,
                            optimizer: Optimizer, optimizer_idx: int) -> None:
        optimizer.zero_grad(set_to_none=True)

    def get_progress_bar_dict(self) -> dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
