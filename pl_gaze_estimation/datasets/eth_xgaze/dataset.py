import json
import pathlib
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig

from ...pl_utils.dataset import Dataset as PlDataset
from ...utils import str2path
from .transforms import create_transform


class OnePersonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: pathlib.Path, transform: Callable):
        self.dataset_path = dataset_path
        self.transform = transform
        self.random_horizontal_flip = False
        self._length = self._get_length()

    def _get_length(self) -> int:
        with h5py.File(self.dataset_path, 'r', swmr=True) as f:
            length = len(f['face_patch'])
        return length

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r', swmr=True) as f:
            image = f['face_patch'][index]
            pose = f['face_head_pose'][index]
            gaze = f['face_gaze'][index]
        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
            pose *= np.array([1, -1])
            gaze *= np.array([1, -1])
        image = self.transform(image)
        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        return image, pose, gaze

    def __len__(self) -> int:
        return self._length


class Dataset(PlDataset):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_root_dir = str2path(self.config.DATASET.DATASET_ROOT_DIR)
        assert dataset_root_dir.exists()

        split_file = dataset_root_dir / 'train_test_split.json'
        with open(split_file) as f:
            split = json.load(f)
        train_paths = [
            dataset_root_dir / 'train' / name for name in split['train']
        ]
        assert len(train_paths) == 80
        for path in train_paths:
            assert path.exists()

        if stage is None or stage == 'fit':
            train_transform = create_transform(self.config, 'train')
            if (self.config.VAL.VAL_RATIO > 0
                    and self.config.VAL.VAL_INDICES is not None):
                raise ValueError
            elif self.config.VAL.VAL_RATIO > 0:
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for path in train_paths
                ])
                val_ratio = self.config.VAL.VAL_RATIO
                assert val_ratio < 1
                val_num = int(len(train_dataset) * val_ratio)
                train_num = len(train_dataset) - val_num
                lengths = [train_num, val_num]
                (self.train_dataset,
                 self.val_dataset) = torch.utils.data.dataset.random_split(
                     train_dataset, lengths)
                val_transform = create_transform(self.config, 'val')
                self.val_dataset.transform = val_transform
            elif self.config.VAL.VAL_INDICES is not None:
                val_indices = set(self.config.VAL.VAL_INDICES)
                assert 0 < len(val_indices) < 80
                for index in val_indices:
                    assert 0 <= index < 80

                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for i, path in enumerate(train_paths)
                    if i not in val_indices
                ])

                val_transform = create_transform(self.config, 'val')
                self.val_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, val_transform)
                    for i, path in enumerate(train_paths) if i in val_indices
                ])
            else:
                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(path, train_transform)
                    for path in train_paths
                ])
            if self.config.DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP:
                for dataset in self.train_dataset.datasets:
                    dataset.random_horizontal_flip = True
