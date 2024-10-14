import argparse
from pathlib import Path

import albumentations as A
import torch
from torch.utils.data import DataLoader
import wandb

from models.unet import UNet
from utils.dataset import ImageDataset


def train(image_dir: str):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str)
    image_dir = parser.parse_args().image_dir
    train(image_dir)

