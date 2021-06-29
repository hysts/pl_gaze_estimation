import numpy as np
from omegaconf import DictConfig


def get_dataset_stats(config: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    mean = np.array(config.DATASET.MEAN)
    std = np.array(config.DATASET.STD)
    return mean, std
