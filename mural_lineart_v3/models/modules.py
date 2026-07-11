"""
基础模块：SelfAttention + ResnetBlock + 条件判别器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """轻量自注意力：Transformer缩放 + FP32 Softmax，防前向NaN"""

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_dim // 8) ** -0.5  # 1/sqrt(d_k) 防溢出

    def forward(self, x):
        B, C, H, W = x.size()
        assert H * W <= 16384, f"SA input too large: {H}x{W}, max 128x128"
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key) * self.scale
        attention = F.softmax(energy.float(), dim=-1).to(energy.dtype)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class GatedConv2d(nn.Module):
    """门控卷积：动态屏蔽 Mask 边界造成的虚假边缘伪影"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.feature_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.gating_conv  = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        nn.init.constant_(self.gating_conv.bias, 2.0)  # 初始gate≈0.88，防梯度消失
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = F.elu(self.feature_conv(x))
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
    """FP32 安全判别器：强制单精度，LSGAN 用"""

    def __init__(self, in_ch=5, base_ch=64):
        super().__init__()
        from torch.nn.utils import spectral_norm as sn
        self.blocks = nn.ModuleList()
        chs = [in_ch, base_ch, 128, 256, 512, 512, 1]
        for i in range(len(chs) - 1):
            stride = 1 if i >= len(chs) - 3 else 2
            use_norm = 1 <= i < len(chs) - 2
            layers = [sn(nn.Conv2d(chs[i], chs[i+1], kernel_size=4, stride=stride, padding=1), eps=1e-4)]
            if use_norm: layers.append(nn.InstanceNorm2d(chs[i+1]))
            layers.append(nn.LeakyReLU(0.2, True) if i < len(chs) - 2 else nn.Identity())
            self.blocks.append(nn.Sequential(*layers) if len(layers) > 1 else layers[0])

    def forward(self, x, return_features=False):
        x = x.float()
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_features and i < len(self.blocks) - 1:
                feats.append(x)
        return (x, feats) if return_features else x


class MultiScaleDiscriminator(nn.Module):
    """Pix2PixHD 风格多尺度判别器：3 个相同架构 D 在不同分辨率工作"""

    def __init__(self, in_ch=5, base_ch=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [Discriminator(in_ch, base_ch) for _ in range(num_scales)]
        )

    def forward(self, x, return_features=False):
        results = []
        for i, d in enumerate(self.discriminators):
            if i == 0:
                x_i = x
            else:
                x_i = nn.functional.interpolate(
                    x, scale_factor=1.0 / (2 ** i),
                    mode="bilinear", align_corners=False
                )
            results.append(d(x_i, return_features=return_features))
        return results  # list of (out, [feats]) per scale


