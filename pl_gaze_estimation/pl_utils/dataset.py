from typing import Optional

import pytorch_lightning as pl
import torch.utils.data
from omegaconf import DictConfig


class Dataset(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.TRAIN.BATCH_SIZE,
            shuffle=self.config.TRAIN.SHUFFLE,
            num_workers=self.config.TRAIN.NUM_WORKERS,
            pin_memory=self.config.TRAIN.PIN_MEMORY,
            drop_last=self.config.TRAIN.DROP_LAST)

    def val_dataloader(self):
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.config.VAL.BATCH_SIZE,
                shuffle=False,
                num_workers=self.config.VAL.NUM_WORKERS,
                pin_memory=self.config.VAL.PIN_MEMORY,
                drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.TEST.NUM_WORKERS,
            pin_memory=self.config.TEST.PIN_MEMORY,
            drop_last=False)
