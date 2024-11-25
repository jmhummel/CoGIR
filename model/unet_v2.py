import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledAttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, scale=16):
        super(PooledAttentionBlock, self).__init__()
        self.scale = scale
        self.norm = torch.nn.GroupNorm(32, in_channels)
        self.qkv = torch.nn.Conv1d(in_channels, in_channels*3, kernel_size=1)
        self.attn = torch.nn.MultiheadAttention(in_channels, 1, batch_first=True)
        self.out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        _input = x
        x = F.avg_pool2d(x, self.scale)  # Downsample using average pooling
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = self.norm(x)
        qkv = self.qkv(x).permute(0, 2, 1)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        x, _ = self.attn(q, k, v, need_weights=False)
        x = x.permute(0, 2, 1)
        x = self.out(x)
        x = x.view(b, c, h, w)
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')  # Upsample using nearest neighbor
        return x + _input

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.norm1 = torch.nn.GroupNorm(32, in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        _input = x
        x = self.norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = torch.nn.functional.silu(x)
        x = self.conv2(x)
        return x + self.shortcut(_input)

class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, model_channels=128, use_attention=True):
        super(UNetV2, self).__init__()

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1),

            ResnetBlock(model_channels, model_channels),
            ResnetBlock(model_channels, model_channels),
            Downsample(model_channels, model_channels),

            nn.Sequential(
                ResnetBlock(model_channels, model_channels*2),
                PooledAttentionBlock(model_channels*2) if use_attention else torch.nn.Identity(),
            ),
            nn.Sequential(
                ResnetBlock(model_channels*2, model_channels*2),
                PooledAttentionBlock(model_channels*2) if use_attention else torch.nn.Identity(),
            ),
            Downsample(model_channels*2, model_channels*2),

            ResnetBlock(model_channels*2, model_channels*2),
            ResnetBlock(model_channels*2, model_channels*2),
            Downsample(model_channels*2, model_channels*2),

            ResnetBlock(model_channels*2, model_channels*2),
            ResnetBlock(model_channels*2, model_channels*2),
        ])

        self.middle_block = nn.Sequential(
            ResnetBlock(model_channels*2, model_channels*2),
            ResnetBlock(model_channels*2, model_channels*2),
        )

        self.output_blocks = torch.nn.ModuleList([
            ResnetBlock(model_channels*4, model_channels*2),
            ResnetBlock(model_channels*4, model_channels*2),
            nn.Sequential(
                ResnetBlock(model_channels*4, model_channels*2),
                Upsample(model_channels*2, model_channels*2),
            ),

            ResnetBlock(model_channels*4, model_channels*2),
            ResnetBlock(model_channels*4, model_channels*2),
            nn.Sequential(
                ResnetBlock(model_channels*4, model_channels*2),
                Upsample(model_channels*2, model_channels*2),
            ),

            ResnetBlock(model_channels*4, model_channels*2),
            ResnetBlock(model_channels*4, model_channels*2),
            nn.Sequential(
                ResnetBlock(model_channels*3, model_channels*2),
                Upsample(model_channels*2, model_channels*2),
            ),

            ResnetBlock(model_channels*3, model_channels),
            ResnetBlock(model_channels*2, model_channels),
            ResnetBlock(model_channels*2, model_channels),
        ])

        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(32, model_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        hs = []
        for module in self.input_blocks:
            x = module(x)
            hs.append(x)
        x = self.middle_block(x)
        for module in self.output_blocks:
            x = torch.cat([x, hs.pop()], 1)
            x = module(x)
        return self.out(x)