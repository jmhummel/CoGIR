import argparse
import logging
from pathlib import Path
from pprint import pformat

import albumentations as A
import hydra
import torch
import torchvision
from torchmetrics.functional import precision

import wandb
from hydra.utils import instantiate
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

BANNER = """
 ██████╗ ██████╗  ██████╗ ██╗██████╗ 
██╔════╝██╔═══██╗██╔════╝ ██║██╔══██╗
██║     ██║   ██║██║  ███╗██║██████╔╝
██║     ██║   ██║██║   ██║██║██╔══██╗
╚██████╗╚██████╔╝╚██████╔╝██║██║  ██║
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚═╝
"""

print(BANNER)


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
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 10 == 0:
            # TODO: show image grid
            # pass
            input_output_target = torch.cat([inputs, outputs, targets], dim=3)
            # grid = torchvision.utils.make_grid(input_output_target, nrow=12)
            # self.log("train/input_output_target", grid)
            # self.logger.log({"train/input_output_target": [wandb.Image(x) for x in input_output_target]})
            self.logger.log_image(key="train/input_output_target", images=[wandb.Image(x) for x in input_output_target])
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # TODO: show image grid
        input_output_target = torch.cat([inputs, outputs, targets], dim=3)
        # grid = torchvision.utils.make_grid(input_output_target, nrow=4)
        # self.log("val/input_output_target", grid)
        # self.logger.log({"val/input_output_target": [wandb.Image(x) for x in input_output_target]})
        self.logger.log_image(key="val/input_output_target", images=[wandb.Image(x) for x in input_output_target])
        return loss

    def configure_optimizers(self):
        return self.optimizer

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    log_config(cfg)

    float32_matmul_precision = cfg.train.get("float32_matmul_precision")
    if float32_matmul_precision:
        torch.set_float32_matmul_precision(float32_matmul_precision)

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
        precision=cfg.train.precision,
        devices=cfg.train.gpus if torch.cuda.is_available() else 0,
        strategy="ddp",
    )

    # Train the model
    trainer.fit(lightning_module, data_module)

    # # Finish the W&B run
    # wandb.finish()

if __name__ == "__main__":
    train()


