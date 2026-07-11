"""Attention Gate — 引导解码器聚焦非连续、非均匀破损区域，确保线条流转自然衔接"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """空间注意力门控：将 skip 特征与上采样特征对齐，生成注意力权重"""

    def __init__(self, F_g, F_l, F_int):
        """
        F_g:   解码器（上采样）特征通道数
        F_l:   编码器（skip）特征通道数
        F_int: 注意力中间通道数
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        """
        g: 解码器上采样特征
        x: 编码器 skip 特征（同分辨率）
        返回: x * alpha（注意力加权后的 skip 特征）, alpha（注意力权重图）
        """
        # 对齐尺寸
        if g.shape[2:] != x.shape[2:]:
            g = nn.functional.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(torch.relu(g1 + x1))
        return x * alpha, alpha
