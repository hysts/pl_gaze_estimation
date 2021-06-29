#!/usr/bin/env python

import logging

import coloredlogs
from pytorch_lightning.utilities.seed import seed_everything

from pl_gaze_estimation.config import parse_args
from pl_gaze_estimation.datasets import create_dataset
from pl_gaze_estimation.models import create_model
from pl_gaze_estimation.pl_utils import get_trainer

coloredlogs.install(level='DEBUG',
                    logger=logging.getLogger('pl_gaze_estimation'))


def main():
    config = parse_args()

    seed_everything(seed=config.EXPERIMENT.SEED, workers=True)
    dataset = create_dataset(config)
    model = create_model(config)
    trainer = get_trainer(config)

    trainer.fit(model, dataset)
    if config.TEST.RUN_TEST:
        trainer.test(ckpt_path=None, verbose=False)


if __name__ == '__main__':
    main()
