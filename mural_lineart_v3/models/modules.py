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


