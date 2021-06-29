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
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.person_id_str = person_id_str
        self.dataset_path = dataset_path
        self.transform = transform
        self.random_horizontal_flip = False

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
        if self.random_horizontal_flip and np.random.rand() < 0.5:
            image = image[:, ::-1]
            pose *= np.array([1, -1])
            gaze *= np.array([1, -1])
        image = self.transform(image)
        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        return image, pose, gaze

    def __len__(self) -> int:
        return 3000


class Dataset(PlDataset):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_path = str2path(self.config.DATASET.DATASET_PATH)
        assert dataset_path.exists()

        assert self.config.EXPERIMENT.TEST_ID in range(-1, 15)
        person_ids = [f'p{index:02}' for index in range(15)]

        if stage is None or stage == 'fit':
            train_transform = create_transform(self.config, 'train')
            if self.config.EXPERIMENT.TEST_ID == -1:
                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_path, train_transform)
                    for person_id in person_ids
                ])
                assert len(self.train_dataset) == 45000
            else:
                test_person_id = person_ids[self.config.EXPERIMENT.TEST_ID]
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_path, train_transform)
                    for person_id in person_ids if person_id != test_person_id
                ])
                assert len(train_dataset) == 42000

                val_transform = create_transform(self.config, 'val')
                if self.config.VAL.USE_TEST_AS_VAL:
                    self.train_dataset = train_dataset
                    self.val_dataset = OnePersonDataset(
                        test_person_id, dataset_path, val_transform)
                else:
                    val_ratio = self.config.VAL.VAL_RATIO
                    assert val_ratio < 1
                    val_num = int(len(train_dataset) * val_ratio)
                    train_num = len(train_dataset) - val_num
                    lengths = [train_num, val_num]
                    self.train_dataset, self.val_dataset = torch.utils.data.dataset.random_split(
                        train_dataset, lengths)
                    self.val_dataset.transform = val_transform
            if self.config.DATASET.TRANSFORM.TRAIN.HORIZONTAL_FLIP:
                for dataset in self.train_dataset.datasets:
                    dataset.random_horizontal_flip = True

        if stage is None or stage == 'test':
            test_transform = create_transform(self.config, 'test')
            test_person_id = person_ids[self.config.EXPERIMENT.TEST_ID]
            self.test_dataset = OnePersonDataset(test_person_id, dataset_path,
                                                 test_transform)
            assert len(self.test_dataset) == 3000
