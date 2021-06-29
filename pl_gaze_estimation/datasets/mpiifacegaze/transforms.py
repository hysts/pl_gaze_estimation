from typing import Callable

import torchvision.transforms as T
from omegaconf import DictConfig

from ..utils import get_dataset_stats


def create_transform(config: DictConfig, stage: str) -> Callable:
    assert stage in ['train', 'val', 'test']

    mean, std = get_dataset_stats(config)
    transform_config = getattr(config.DATASET.TRANSFORM, stage.upper())

    transforms = [
        T.Lambda(lambda x: x[:, :, ::-1]),  # BGR -> RGB
        T.ToPILImage(),
    ]
    if 'RESIZE' in transform_config and transform_config.RESIZE != 448:
        transforms.append(T.Resize(transform_config.RESIZE))
    if 'COLOR_JITTER' in transform_config:
        transforms.append(
            T.ColorJitter(brightness=transform_config.COLOR_JITTER.BRIGHTNESS,
                          contrast=transform_config.COLOR_JITTER.CONTRAST,
                          saturation=transform_config.COLOR_JITTER.SATURATION,
                          hue=transform_config.COLOR_JITTER.HUE))

    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean, std, inplace=False),
    ])
    return T.Compose(transforms)
