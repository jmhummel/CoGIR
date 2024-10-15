import pytest
import numpy as np

from pathlib import Path
import albumentations as A
import torch

from data.dataset import ImageDataset


# Test ImageDataset conversion
def test_image_conversion():
    # Dummy dataset
    dataset = ImageDataset(location=Path('./tests/data'), size=(256, 256))

    # Get an item from the dataset (first image)
    input_tensor, target_tensor = dataset[0]

    # Check the dtype
    assert input_tensor.dtype == torch.float32, "Input tensor should be float32"
    assert target_tensor.dtype == torch.float32, "Target tensor should be float32"

    # Check the shape (3 channels, height, width)
    assert input_tensor.shape == (3, 256, 256), "Input tensor shape should be (3, 256, 256)"
    assert target_tensor.shape == (3, 256, 256), "Target tensor shape should be (3, 256, 256)"

    # Check the value range
    assert input_tensor.max() <= 1.0, "Input tensor values should be <= 1.0"
    assert input_tensor.min() >= 0.0, "Input tensor values should be >= 0.0"
    assert target_tensor.max() <= 1.0, "Target tensor values should be <= 1.0"
    assert target_tensor.min() >= 0.0, "Target tensor values should be >= 0.0"
