"""
基础模块：ResnetBlock + 条件判别器
"""

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.conv(x)


class Discriminator(nn.Module):
    """SN-PatchGAN 条件判别器"""

    def __init__(self, in_ch=2, base_ch=64):
        super().__init__()
        from torch.nn.utils import spectral_norm as sn

        self.blocks = nn.ModuleList()
        chs = [in_ch, base_ch, 128, 256, 512, 512, 1]
        for i in range(len(chs) - 1):
            stride = 1 if i >= len(chs) - 3 else 2
            use_norm = 1 <= i < len(chs) - 2
            layers = [sn(nn.Conv2d(chs[i], chs[i+1], kernel_size=4, stride=stride, padding=1))]
            if use_norm: layers.append(nn.InstanceNorm2d(chs[i+1]))
            layers.append(nn.LeakyReLU(0.2, True) if i < len(chs) - 2 else nn.Identity())
            self.blocks.append(nn.Sequential(*layers) if len(layers) > 1 else layers[0])

    def forward(self, x, return_features=False):
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_features and i < len(self.blocks) - 1:
                feats.append(x)
        return (x, feats) if return_features else x
