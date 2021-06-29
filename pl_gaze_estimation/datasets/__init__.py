from typing import Type

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule


def create_dataset(config: DictConfig) -> pl.LightningDataModule:
    Dataset = create_dataset_module(config.DATASET.NAME)
    return Dataset(config)


def create_dataset_module(dataset_name: str) -> Type[LightningDataModule]:
    if dataset_name == 'MPIIGaze':
        from .mpiigaze.dataset import Dataset as MPIIGazeDataset
        return MPIIGazeDataset
    elif dataset_name == 'MPIIFaceGaze':
        from .mpiifacegaze.dataset import Dataset as MPIIFaceGazeDataset
        return MPIIFaceGazeDataset
    elif dataset_name == 'ETH-XGaze':
        from .eth_xgaze.dataset import Dataset as ETHXGazeDataset
        return ETHXGazeDataset
    else:
        raise ValueError
