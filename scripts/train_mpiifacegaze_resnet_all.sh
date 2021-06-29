#!/usr/bin/env bash

set -Ceu

for test_id in {0..14}; do
    python train.py \
        --configs \
            configs/base.yaml \
            configs/model/mpiifacegaze/resnet_simple.yaml \
            configs/dataset/mpiifacegaze.yaml \
            configs/optimizer/sgd.yaml \
            configs/scheduler/multistep.yaml \
            configs/scheduler/warmup.yaml \
        --options \
            SCHEDULER.EPOCHS 15 \
            SCHEDULER.MULTISTEP.MILESTONES '[10, 13]' \
            SCHEDULER.WARMUP.EPOCHS 3 \
            TRAIN.BATCH_SIZE 32 \
            VAL.BATCH_SIZE 256 \
            OPTIMIZER.LR 0.1 \
            LOG.WANDB.PROJECT pl_mpiifacegaze \
            LOG.CHECKPOINT.PERIOD 5 \
            EXPERIMENT.ROOT_DIR experiments/mpiifacegaze \
            EXPERIMENT.TEST_ID ${test_id} \
            EXPERIMENT.OUTPUT_DIR exp0000/$(printf %02d ${test_id})
done
