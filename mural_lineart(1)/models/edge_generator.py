"""
EdgeConnect 风格网络：EdgeGenerator + Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EdgeGenerator(nn.Module):
    """EdgeConnect Edge Generator: Encoder → ResBlocks → Decoder"""

    def __init__(self, in_ch=3, out_ch=1, residual_blocks=8, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_ch, kernel_size=7),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4),
            nn.ReLU(True),
        )

        # Residual blocks
        blocks = []
        for _ in range(residual_blocks):
            blocks.append(ResnetBlock(base_ch * 4))
        self.resblocks = nn.Sequential(*blocks)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
        )
        self.dec3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, out_ch, kernel_size=7),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.resblocks(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        return x


class Discriminator(nn.Module):
    """SN-PatchGAN Discriminator: 1→64→128→256→512(stride=1)→1"""

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
