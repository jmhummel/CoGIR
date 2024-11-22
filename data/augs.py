import numpy as np
from albumentations import ImageOnlyTransform


class ClipMinToZero(ImageOnlyTransform):
    """
    Custom augmentation to clip pixel values to ensure a minimum of 0.
    Works for both uint8 ([0, 255]) and float32 ([0, 1]) images.
    """
    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        # Clip all pixel values to a minimum of 0
        return np.clip(img, 0, None)

    def get_transform_init_args_names(self):
        return []



class CenterSquareCrop(ImageOnlyTransform):
    """
    Custom Albumentation augmentation to crop the center square of an image.
    """
    def __init__(self, always_apply=True, p=1.0):
        super(CenterSquareCrop, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # Determine the smaller dimension to define the square crop size
        height, width = image.shape[:2]
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        return image[start_y:start_y + crop_size, start_x:start_x + crop_size]

    def get_transform_init_args_names(self):
        return []