import pathlib
from typing import Callable, Optional, Tuple

import h5py
import torch
import torch.utils.data
from omegaconf import DictConfig

from ...pl_utils.dataset import Dataset as PlDataset
from ...utils import str2path
from .transforms import create_transform


class OnePersonDataset(torch.utils.data.Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        with h5py.File(dataset_path, 'r') as f:
            images = f.get(f'{person_id_str}/image')[()]
            poses = f.get(f'{person_id_str}/pose')[()]
            gazes = f.get(f'{person_id_str}/gaze')[()]
        assert len(images) == 3000
        assert len(poses) == 3000
        assert len(gazes) == 3000
        self.images = images
        self.poses = poses
        self.gazes = gazes

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        gaze = torch.from_numpy(self.gazes[index])
        return image, pose, gaze

    def __len__(self) -> int:
        return len(self.images)


class Dataset(PlDataset):
    def __init__(self, config: DictConfig):
        super().__init__(config)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_path = str2path(self.config.DATASET.DATASET_PATH)
        assert dataset_path.exists()

        assert self.config.EXPERIMENT.TEST_ID in range(-1, 15)
        person_ids = [f'p{index:02}' for index in range(15)]

        transform = create_transform()

        if stage is None or stage == 'fit':
            if self.config.EXPERIMENT.TEST_ID == -1:
                self.train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_path, transform)
                    for person_id in person_ids
                ])
                assert len(self.train_dataset) == 45000
            else:
                test_person_id = person_ids[self.config.EXPERIMENT.TEST_ID]
                train_dataset = torch.utils.data.ConcatDataset([
                    OnePersonDataset(person_id, dataset_path, transform)
                    for person_id in person_ids if person_id != test_person_id
                ])
                assert len(train_dataset) == 42000

                if self.config.VAL.USE_TEST_AS_VAL:
                    self.train_dataset = train_dataset
                    self.val_dataset = OnePersonDataset(
                        test_person_id, dataset_path, transform)
                else:
                    val_ratio = self.config.VAL.VAL_RATIO
                    assert val_ratio < 1
                    val_num = int(len(train_dataset) * val_ratio)
                    train_num = len(train_dataset) - val_num
                    lengths = [train_num, val_num]
                    self.train_dataset, self.val_dataset = torch.utils.data.dataset.random_split(
                        train_dataset, lengths)

        if stage is None or stage == 'test':
            test_person_id = person_ids[self.config.EXPERIMENT.TEST_ID]
            self.test_dataset = OnePersonDataset(test_person_id, dataset_path,
                                                 transform)
            assert len(self.test_dataset) == 3000
