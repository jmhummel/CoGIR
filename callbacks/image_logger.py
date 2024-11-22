# This file is based on the original work from the ControlNet repository.
#
# Original file:
#   - File name and path: cldm/logger.py
#   - Original URL: https://github.com/lllyasviel/ControlNet/blob/main/cldm/logger.py
#   - Original commit hash: e38d22aa1ce2c2c72d2536c8f337b47249033c98 (Feb 10, 2023)
#
# Original repository:
#   - Repository name: ControlNet
#   - Repository URL: https://github.com/lllyasviel/ControlNet
#
# Author of the original file:
#   - User ID: lllyasviel
#   - Name: Lvmin Zhang
#
# License:
#   This file is licensed under the Apache License, Version 2.0.
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications:
#   This file has been modified from its original version. The modifications
#   include the addition of Wandb logging functionality.

import os
from pathlib import Path
import datetime

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.loggers import WandbLogger


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.save_dir = Path("outputs") / datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_wandb(self, pl_module, images, global_step):
        if not isinstance(pl_module.logger, WandbLogger):
            return

        # Concatenate images for WandB logging
        concatenated_images = []
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)  # Arrange in a single column
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            concatenated_images.append(Image.fromarray(grid))

        # Combine images vertically
        total_height = sum(img.height for img in concatenated_images)
        max_width = max(img.width for img in concatenated_images)
        combined_image = Image.new("RGB", (max_width, total_height))

        y_offset = 0
        for img in concatenated_images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height

        pl_module.logger.log_image(key="Target/Input/Prompt/Output", images=[combined_image])

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)


            self.log_local(self.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
            self.log_wandb(pl_module, images, pl_module.global_step)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
