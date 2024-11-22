import argparse
import logging
from pathlib import Path
from pprint import pformat
from typing import Literal

import albumentations as A
import hydra
import torch
import torchvision
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.checkpoint import checkpoint
from torchmetrics.functional import precision

import wandb
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from ldm.util import instantiate_from_config
from cldm.model import load_state_dict

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
    def __init__(self, model, optimizer, criterion, stage: Literal["latent", "pixel"] = "pixel"):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.stage = stage
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        if self.stage == "latent":
            # Minimize the latent space loss
            with torch.no_grad():
                combined = torch.cat([inputs, targets], dim=0)
                posterior = self.model.autoencoder.encode(combined)
                z = posterior.sample()
                z_inputs, z_targets = z.chunk(2, dim=0)
            z_outputs = self.model.unet(z_inputs)
            loss = self.criterion(z_outputs, z_targets)
            outputs = None
        elif self.stage == "pixel":
            # Minimize the reconstruction loss
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            z_outputs = None
        else:
            raise ValueError(f"Invalid stage: {self.stage}")
        # outputs = self.model(inputs)
        # loss = self.criterion(outputs, targets)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx % 10 == 0:
            # TODO: show image grid
            # pass
            if outputs is None and z_outputs is not None:
                # use the z_outputs to generate the images, we don't need the computation graph
                outputs = self.model.autoencoder.decode(z_outputs)
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

def load_autoencoder(controlnet_weights):
    cfg = OmegaConf.load('config/model/cldm_v15.yaml')
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model = instantiate_from_config(model_config)
    model.load_state_dict(load_state_dict(controlnet_weights, location='cpu'), strict=False)
    autoencoder = model.first_stage_model
    autoencoder.eval()
    # freeze the weights
    for param in autoencoder.parameters():
        param.requires_grad = False
    return autoencoder

def load_unet_weights(unet, controlnet_weights):
    state_dict = load_state_dict(controlnet_weights, location='cpu')

    state_dict_renamed_keys = {}
    for k, v in state_dict.items():
        if k not in unet.state_dict():
            continue
        if k.startswith('model.diffusion_model.'):
            k = k.replace('model.diffusion_model.', '')
            state_dict_renamed_keys[k] = v
    for k, v in state_dict_renamed_keys.items():
        if k.endswith('proj_out.weight'):
            state_dict_renamed_keys[k] = v.squeeze(-1)
    for k, v in unet.state_dict().items():
        if k not in state_dict_renamed_keys:
            state_dict_renamed_keys[k] = v
    unet.load_state_dict(state_dict_renamed_keys, strict=False)

class LatentUnet(torch.nn.Module):
    def __init__(self, autoencoder, unet):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet

    def forward(self, x):
        with torch.no_grad():
            posterior = self.autoencoder.encode(x)
            z = posterior.sample()
        pred_z = self.unet(z)
        dec = self.autoencoder.decode(pred_z)
        return dec

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    log_config(cfg)

    float32_matmul_precision = cfg.train.get("float32_matmul_precision")
    if float32_matmul_precision:
        torch.set_float32_matmul_precision(float32_matmul_precision)

    # Instantiate all the objects using Hydra
    logger = instantiate(cfg.logger)
    data_module = instantiate(cfg.data)
    autoencoder = load_autoencoder(cfg.train.controlnet_weights)
    unet = instantiate(cfg.model)
    # load_unet_weights(unet, cfg.train.controlnet_weights)
    model = LatentUnet(autoencoder, unet)
    optimizer = instantiate(cfg.optimizer, params=model.unet.parameters())
    criterion = instantiate(cfg.criterion)
    callbacks = [instantiate(c) for c in cfg.callbacks.values()]
    stage = cfg.train.stage

    # Create the LightningModule
    lightning_module = LightningModule(model, optimizer, criterion, stage=stage)

    # Create the Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        min_epochs=cfg.train.epochs,
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        devices=cfg.train.gpus if torch.cuda.is_available() else 0,
        strategy="ddp",
        accelerator="gpu",
    )

    # Train the model
    trainer.fit(lightning_module, data_module)

    # # Finish the W&B run
    # wandb.finish()

if __name__ == "__main__":
    train()


