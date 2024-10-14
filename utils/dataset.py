from glob import glob
from pathlib import Path
from typing import Union, Tuple, List

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def get_image_paths(image_dir: Path) -> List[Path]:
    return list(image_dir.rglob('*.jpg')) + list(image_dir.rglob('*.png'))

class ImageDataset(Dataset):
    def __init__(self,
                 location: Union[str, Path],
                 size: Union[int, Tuple[int, int]] = (512, 512),
                 augmentations: List[A.BasicTransform | A.BaseCompose] = (),
                 degradations: List[A.BasicTransform | A.BaseCompose] = (),
                 **kwargs):
        self.location = Path(location)
        self.paths = get_image_paths(self.location)

        if isinstance(size, int):
            size = (size, size)
        self.size = size

        self.resize = A.Resize(*size, interpolation=cv2.INTER_CUBIC)
        self.augmentations = A.Compose(augmentations)
        self.degradations = A.Compose(degradations)
        self.to_tensor = ToTensorV2()


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0


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
