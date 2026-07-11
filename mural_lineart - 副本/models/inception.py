"""Inception 多尺度卷积模块 — 平衡噪点抑制与线条保真"""

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """四分支并行多尺度特征提取，不同感受野覆盖细线～粗结构"""

    def __init__(self, in_ch, out_ch, ratios=None):
        """
        ratios: 4 元素列表，控制 1×1/3×3/5×5/pool 四条分支的输出通道比例
                默认 [0.25, 0.25, 0.25, 0.25]，和 = 1.0
        """
        super().__init__()
        if ratios is None:
            ratios = [0.25, 0.25, 0.25, 0.25]

        c1, c2, c3, c4 = [max(1, int(out_ch * r)) for r in ratios]
        # 弥补取整误差，确保 concat 后 = out_ch
        c1 += out_ch - (c1 + c2 + c3 + c4)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # 两个级联 3×3 代替 5×5（参数更少，感受野等效）
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, c3, kernel_size=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, c4, kernel_size=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)
