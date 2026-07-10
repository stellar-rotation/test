"""
Skeleton Loss — 软骨架化 + L1，强制线条拓扑连续性
断线在骨架上表现为拓扑断裂，此 Loss 对此极度敏感
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """可微骨架化：maxpool → minpool 迭代逼近"""

    def __init__(self, iterations=5):
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        s = x.clone()
        for _ in range(self.iterations):
            s = -F.max_pool2d(-s, kernel_size=3, stride=1, padding=1)
            s = F.max_pool2d(s, kernel_size=3, stride=1, padding=1)
        return s


class SkeletonLoss(nn.Module):
    """对预测和 GT 分别软骨架化后计算 L1"""

    def __init__(self, iterations=5, input_range="tanh"):
        """
        input_range: 'tanh' ([-1,1]) or 'sigmoid' ([0,1])
        """
        super().__init__()
        self.skel = SoftSkeletonize(iterations)
        self.l1 = nn.L1Loss()
        self.input_range = input_range

    def forward(self, pred, target):
        if self.input_range == "tanh":
            pred_01 = (pred + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0
        else:  # sigmoid [0,1]
            pred_01 = pred
            target_01 = target

        tau, T = 0.5, 0.1
        pred_sharp = torch.sigmoid((pred_01 - tau) / T)
        target_sharp = torch.sigmoid((target_01 - tau) / T)

        skel_pred = self.skel(pred_sharp)
        skel_target = self.skel(target_sharp)
        return self.l1(skel_pred, skel_target)
