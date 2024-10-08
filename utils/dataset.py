from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.paths = glob(image_dir + '/*.jpg')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')

        # Apply augmentations to generate input image
        if self.transform:
            input_image = self.transform(image=np.array(image))['image']
        else:
            input_image = np.array(image)

        target_image = np.array(image)

        return input_image, target_image