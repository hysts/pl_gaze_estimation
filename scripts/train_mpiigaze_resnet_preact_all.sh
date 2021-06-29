#!/usr/bin/env bash

set -Ceu

for test_id in {0..14}; do
    python train.py \
        --configs \
            configs/base.yaml \
            configs/model/mpiigaze/resnet_preact.yaml \
            configs/dataset/mpiigaze.yaml \
            configs/optimizer/sgd.yaml \
            configs/scheduler/multistep.yaml \
        --options \
            SCHEDULER.EPOCHS 40 \
            SCHEDULER.MULTISTEP.MILESTONES '[30, 35]' \
            TRAIN.BATCH_SIZE 32 \
            VAL.BATCH_SIZE 256 \
            OPTIMIZER.LR 0.1 \
            LOG.WANDB.PROJECT pl_mpiigaze \
            EXPERIMENT.ROOT_DIR experiments/mpiigaze \
            EXPERIMENT.TEST_ID ${test_id} \
            EXPERIMENT.OUTPUT_DIR exp0000/$(printf %02d ${test_id})
done
