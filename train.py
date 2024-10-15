import argparse
import logging
from pathlib import Path
from pprint import pformat

import albumentations as A
import hydra
import torch
from hydra.utils import instantiate
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import wandb

from model.unet import UNet
from data.dataset import ImageDataset

log = logging.getLogger(__name__)

@rank_zero_only
def log_config(cfg: DictConfig):
    log.info('-' * 30)
    log.info(pformat(OmegaConf.to_container(cfg, resolve=True)))
    log.info('-' * 30)

class LightningModule(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    log_config(cfg)

    # Instantiate all the objects using Hydra
    logger = instantiate(cfg.logger)
    data_module = instantiate(cfg.data)
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = instantiate(cfg.criterion)
    callbacks = [instantiate(c) for c in cfg.callbacks.values()]

    # Create the LightningModule
    lightning_module = LightningModule(model, optimizer, criterion)

    # Create the Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=cfg.train.epochs,
        devices=cfg.train.gpus if torch.cuda.is_available() else 0,
    )

    # Train the model
    trainer.fit(lightning_module, data_module)

    # Finish the W&B run
    wandb.finish()

if __name__ == "__main__":
    train()

"""
def _train(image_dir: str):
    wandb.init(project="CoGIR")

    dataset = ImageDataset(location=Path(image_dir), size=(512, 512),
                           degradations=[A.RandomBrightnessContrast(p=1.0, brightness_limit=0.5, contrast_limit=0.5),])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet(in_channels=3, out_channels=3)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(10):
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{10}], Loss: {loss.item():.4f}")
            wandb.log({"loss": loss.item()})

            if i % 10 == 0:
                input_output_target = torch.cat([inputs, outputs, targets], dim=3)
                wandb.log({"input_output_target": [wandb.Image(x) for x in input_output_target]})
"""

