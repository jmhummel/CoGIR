import torch
from lpips import LPIPS
from torch import nn
import torch.nn.functional as F


class LPIPSLoss(nn.Module):
    def __init__(self, use_l1: bool = False):
        super().__init__()
        self.lpips = LPIPS(net='vgg')
        self.use_l1 = use_l1

    def forward(self, x, y):
        with torch.no_grad():
            loss = self.lpips(x, y, normalize=True).mean()
            if self.use_l1:
                loss += F.l1_loss(x, y)
            return loss