"""
基础模块：SelfAttention + ResnetBlock + 条件判别器
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """轻量自注意力：全局感受野，不撕裂像素"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class GatedConv2d(nn.Module):
    """门控卷积：动态屏蔽 Mask 边界造成的虚假边缘伪影"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.gating_conv  = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.feature_conv(x)
        gating = self.sigmoid(self.gating_conv(x))
        return feature * gating


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
