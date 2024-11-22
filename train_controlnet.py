import argparse
import datetime
import logging
from pathlib import Path
from pprint import pformat

import albumentations as A
import hydra
import torch
import torchvision
from cldm.logger import ImageLogger
from ldm.util import instantiate_from_config
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics.functional import precision

import wandb
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

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


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    log_config(cfg)

    float32_matmul_precision = cfg.train.get("float32_matmul_precision")
    if float32_matmul_precision:
        torch.set_float32_matmul_precision(float32_matmul_precision)

    # Instantiate all the objects using Hydra
    logger = instantiate(cfg.logger)
    # logger.save_dir = Path("outputs") / datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_module = instantiate(cfg.data)
    callbacks = [instantiate(c) for c in cfg.callbacks.values()]
    # image_logger = ImageLogger(batch_frequency=200)
    # callbacks.append(image_logger)

    # Create the LightningModule
    model_config = OmegaConf.to_container(cfg.model.model, resolve=True)
    lightning_module = instantiate_from_config(model_config)
    lightning_module.load_state_dict(load_state_dict(cfg.train.resume_path, location='cpu'), strict=False)
    lightning_module.learning_rate = cfg.train.learning_rate
    lightning_module.sd_locked = cfg.train.sd_locked
    lightning_module.only_mid_control = cfg.train.only_mid_control
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
        val_check_interval=1000,
        # check_val_every_n_epoch=None,
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(lightning_module, data_module)

    # # Finish the W&B run
    # wandb.finish()

if __name__ == "__main__":
    train()


