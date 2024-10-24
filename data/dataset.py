from pathlib import Path
from typing import Union, Tuple, List

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, paths: List[Path], **kwargs):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image

class ImageDataset(BaseDataset):
    def __init__(self,
                 paths: List[Path],
                 size: Union[int, Tuple[int, int]] = (512, 512),
                 augmentations: List[A.BasicTransform | A.BaseCompose] = (),
                 degradations: List[A.BasicTransform | A.BaseCompose] = (),
                 **kwargs):
        super().__init__(paths, **kwargs)
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        self.resize = A.Resize(*size, interpolation=cv2.INTER_CUBIC)
        self.augmentations = A.Compose(augmentations)
        self.degradations = A.Compose(degradations)
        self.to_tensor = ToTensorV2()

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        augmented_image = self.augmentations(image=image)['image']
        degraded_image = self.degradations(image=augmented_image)['image']
        input_image = self.resize(image=degraded_image)['image']
        target_image = self.resize(image=augmented_image)['image']

        # Clip the values to be between 0 and 1
        input_image = np.clip(input_image, 0.0, 1.0)
        target_image = np.clip(target_image, 0.0, 1.0)

        input_tensor = self.to_tensor(image=input_image)['image']
        target_tensor = self.to_tensor(image=target_image)['image']
        return input_tensor, target_tensor

class ValidationDataset(BaseDataset):
    def __init__(self, paths: List[Path],
                 size: Union[int, Tuple[int, int]] = (512, 512),
                 # item_specific_degradations: List[List[A.BasicTransform | A.BaseCompose]] = (),
                 **kwargs):
        super().__init__(paths, **kwargs)
        if isinstance(size, int):
            size = (size, size)
        self.size = size

        self.resize = A.Resize(*size, interpolation=cv2.INTER_CUBIC)
        # self.item_specific_degradations = [A.Compose(degradations) for degradations in item_specific_degradations]
        self.to_tensor = ToTensorV2()

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        # degradation = self.item_specific_degradations[idx % len(self.item_specific_degradations)]
        # degraded_image = degradation(image=image)['image']
        # input_image = self.resize(image=degraded_image)['image']
        input_image = self.resize(image=image)['image']
        target_image = self.resize(image=image)['image']

        # Clip the values to be between 0 and 1
        input_image = np.clip(input_image, 0.0, 1.0)
        target_image = np.clip(target_image, 0.0, 1.0)

        input_tensor = self.to_tensor(image=input_image)['image']
        target_tensor = self.to_tensor(image=target_image)['image']
        return input_tensor, target_tensor

