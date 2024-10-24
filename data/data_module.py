from pathlib import Path
from typing import Union

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.utils import get_image_paths, PartialDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self,
                 location: Union[str, Path],
                 train_dataset: PartialDataset,
                 val_dataset: PartialDataset,
                 val_size: int = 32,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = False
                 ):
        super().__init__()
        self.location = location
        self.paths = get_image_paths(location)
        self.train_dataset_partial = train_dataset
        self.val_dataset_partial = val_dataset
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        print(f"Full dataset size: {len(self.paths)}")
        train_size = len(self.paths) - self.val_size
        print(f"Train dataset size: {train_size}")
        print(f"Val dataset size: {self.val_size}")
        train_paths, val_paths = train_test_split(self.paths, test_size=self.val_size, random_state=42)
        self.train_dataset = self.train_dataset_partial(train_paths)
        self.val_dataset = self.val_dataset_partial(val_paths)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, shuffle=False)