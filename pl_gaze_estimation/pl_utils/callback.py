import pathlib
import time
from typing import Any, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ProgressBar
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT


def get_callbacks(config: DictConfig,
                  output_dir: pathlib.Path) -> list[Callback]:
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=config.LOG.CHECKPOINT.FILENAME,
            save_top_k=config.LOG.CHECKPOINT.TOP_K,
            save_last=config.LOG.CHECKPOINT.SAVE_LAST,
            every_n_val_epochs=config.LOG.CHECKPOINT.PERIOD,
            verbose=config.LOG.CHECKPOINT.VERBOSE),
        SaveConfigCallback(config, output_dir),
    ]
    if not config.LOG.SHOW_PROGRESS_BAR:
        callbacks.append(ProgressBar(refresh_rate=0))
    if config.LOG.LOG_ETA:
        callbacks.append(ETACallback())

    return callbacks


class SaveConfigCallback(Callback):
    def __init__(self, config: DictConfig, output_dir: pathlib.Path):
        self.config = config
        self.output_dir = output_dir

    @rank_zero_only
    def on_train_start(self, trainer: 'pl.Trainer',
                       pl_module: 'pl.LightningModule') -> None:
        if not self.config.EXPERIMENT.OUTPUT_DIR:
            return

        config_path = self.output_dir / 'config.yaml'
        if config_path.exists():
            return

        self.output_dir.mkdir(exist_ok=True, parents=True)
        with open(config_path, 'w') as f:
            f.write(
                OmegaConf.to_yaml(self.config, resolve=True, sort_keys=True))


class ETACallback(Callback):
    def __init__(self):
        super().__init__()
        self._trainer = None
        self.train_elapsed = 0.
        self.train_iterations = 0
        self.remaining_train_iterations = 0
        self.train_eta = 0
        self.val_elapsed = 0.
        self.val_iterations = 0
        self.remaining_val_iterations = 0
        self.val_eta = 0
        self._tic = 0.

    @property
    def trainer(self):
        return self._trainer

    @property
    def total_train_batches(self) -> int:
        return self.trainer.num_training_batches

    @property
    def total_val_batches(self) -> int:
        total_val_batches = 0
        if self.trainer.enable_validation:
            is_val_epoch = (self.trainer.current_epoch +
                            1) % self.trainer.check_val_every_n_epoch == 0
            total_val_batches = sum(
                self.trainer.num_val_batches) if is_val_epoch else 0
        return total_val_batches

    @rank_zero_only
    def on_init_end(self, trainer: 'pl.Trainer') -> None:
        self._trainer = trainer

    @rank_zero_only
    def on_train_start(self, trainer: 'pl.Trainer',
                       pl_module: 'pl.LightningModule') -> None:
        max_epochs = self.trainer.max_epochs
        current_epoch = self.trainer.current_epoch
        remaining_epoch = max_epochs - current_epoch
        self.remaining_train_iterations = remaining_epoch * self.total_train_batches

    @rank_zero_only
    def on_train_epoch_start(self, trainer: 'pl.Trainer',
                             pl_module: 'pl.LightningModule') -> None:
        self._tic = time.time()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        elapsed = self.train_elapsed + time.time() - self._tic
        self.train_iterations += 1
        self.remaining_train_iterations -= 1
        if self._should_log(trainer):
            speed = self.train_iterations / elapsed
            self.train_eta = round(self.remaining_train_iterations / speed)
            epoch_eta = (self.remaining_train_iterations %
                         self.total_train_batches) / speed
            trainer.lightning_module.log_dict(
                {
                    'time/eta': self.train_eta + self.val_eta,
                    'time/epoch_eta': epoch_eta,
                    'time/train_batch_index': batch_idx,
                    'time/train_total_batches': self.total_train_batches,
                },
                on_step=True,
                on_epoch=False)
            trainer.lightning_module.log_dict({'time/train_speed': speed},
                                              on_step=True,
                                              on_epoch=False)

    @rank_zero_only
    def on_train_epoch_end(self,
                           trainer: 'pl.Trainer',
                           pl_module: 'pl.LightningModule',
                           unused: Optional[Any] = None) -> None:
        self.train_elapsed += time.time() - self._tic

    @rank_zero_only
    def on_validation_start(self, trainer: 'pl.Trainer',
                            pl_module: 'pl.LightningModule') -> None:
        max_epochs = self.trainer.max_epochs
        current_epoch = self.trainer.current_epoch
        remaining_epoch = max_epochs - current_epoch
        self.remaining_val_iterations = remaining_epoch * self.total_val_batches

    @rank_zero_only
    def on_validation_epoch_start(self, trainer: 'pl.Trainer',
                                  pl_module: 'pl.LightningModule') -> None:
        self._tic = time.time()

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        elapsed = self.val_elapsed + time.time() - self._tic
        self.val_iterations += 1
        self.remaining_val_iterations -= 1
        speed = self.val_iterations / elapsed
        self.val_eta = round(self.remaining_val_iterations / speed)
        trainer.lightning_module.log_dict({'time/val_speed': speed},
                                          on_step=False,
                                          on_epoch=True)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: 'pl.Trainer',
                                pl_module: 'pl.LightningModule') -> None:
        self.val_elapsed += time.time() - self._tic

    @staticmethod
    def _should_log(trainer) -> bool:
        return (trainer.global_step +
                1) % trainer.log_every_n_steps == 0 or trainer.should_stop
