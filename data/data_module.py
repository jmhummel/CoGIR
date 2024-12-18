import pytorch_lightning as pl
import torch
from jupyter_core.version import parts
from torch.utils.data import Dataset, random_split, DataLoader

from data.dataset import ImageDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset: ImageDataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 val_size: int = 32,
                 pin_memory: bool = True,
                 persistent_workers: bool = False
                 ):
        super().__init__()
        self.full_dataset = dataset
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        print(f"Full dataset size: {len(self.full_dataset)}")
        # Assuming full dataset size is greater than val_size
        assert self.val_size < len(self.full_dataset), "Validation size must be less than total dataset size."

        train_size = len(self.full_dataset) - self.val_size
        print(f"Train dataset size: {train_size}")
        print(f"Val dataset size: {self.val_size}")
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, self.val_size],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers, shuffle=False)