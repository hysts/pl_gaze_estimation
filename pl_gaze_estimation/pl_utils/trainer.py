import logging
import pathlib

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from .callback import get_callbacks
from .logger import get_loggers

logger = logging.getLogger(__name__)


def get_trainer(config):
    exp_root_dir = pathlib.Path(config.EXPERIMENT.ROOT_DIR).expanduser()
    exp_root_dir.mkdir(exist_ok=True, parents=True)

    if config.EXPERIMENT.RESUME:
        output_dir = exp_root_dir / config.EXPERIMENT.OUTPUT_DIR
    elif config.EXPERIMENT.OUTPUT_DIR:
        output_dir = exp_root_dir / config.EXPERIMENT.OUTPUT_DIR
        assert not output_dir.exists()
    else:
        output_dir = None
        logger.warning('No checkpoint will be saved.')

    trainer = pl.Trainer(
        max_epochs=config.SCHEDULER.EPOCHS,
        default_root_dir=config.EXPERIMENT.ROOT_DIR,
        accumulate_grad_batches=config.TRAIN.BATCH_ACCUMULATION,
        tpu_cores=config.DEVICE.TPU_CORES,
        gpus=config.DEVICE.GPUS,
        accelerator=config.DEVICE.ACCELERATOR,
        sync_batchnorm=config.DEVICE.SYNC_BN,
        precision=config.DEVICE.PRECISION,
        deterministic=config.DEVICE.CUDNN.DETERMINISTIC,
        benchmark=config.DEVICE.CUDNN.BENCHMARK,
        progress_bar_refresh_rate=config.LOG.PROGRESS_BAR_REFRESH_RATE,
        log_every_n_steps=config.LOG.LOG_PERIOD,
        check_val_every_n_epoch=config.EXPERIMENT.VAL_PERIOD,
        weights_save_path=None,
        resume_from_checkpoint=config.EXPERIMENT.RESUME,
        weights_summary=config.DEBUG.WEIGHT_SUMMARY,
        profiler=config.DEBUG.PROFILER,
        num_sanity_val_steps=config.DEBUG.NUM_SANITY_VAL_STEPS,
        fast_dev_run=config.DEBUG.FAST_DEV_RUN
        if config.DEBUG.DEBUG else False,
        logger=get_loggers(config, exp_root_dir),
        callbacks=get_callbacks(config, output_dir),
        plugins=[
            DDPPlugin(find_unused_parameters=False,
                      sync_batchnorm=config.DEVICE.SYNC_BN)
        ] if config.DEVICE.GPUS > 1 and config.DEVICE.ACCELERATOR == 'ddp'
        and config.DEVICE.USE_DDPPLUGIN else None,
    )
    return trainer
