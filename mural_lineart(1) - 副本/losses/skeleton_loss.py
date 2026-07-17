"""Soft-clDice topology loss for black line art on a white background."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """Differentiable morphological skeletonization."""

    def __init__(self, iterations=10):
        super().__init__()
        self.iterations = iterations

    @staticmethod
    def _erode(x):
        vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
        horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
        return torch.minimum(vertical, horizontal)

    @staticmethod
    def _dilate(x):
        return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    def _open(self, x):
        return self._dilate(self._erode(x))

    def forward(self, x):
        opened = self._open(x)
        skeleton = F.relu(x - opened)
        for _ in range(self.iterations):
            x = self._erode(x)
            delta = F.relu(x - self._open(x))
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return skeleton


class SkeletonLoss(nn.Module):
    """Topology overlap between predicted and target line skeletons."""

    def __init__(self, iterations=10, smooth=1.0):
        super().__init__()
        self.skel = SoftSkeletonize(iterations)
        self.smooth = smooth

    def forward(self, pred, target):
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0

        # Dataset edges are black on white, so invert them to foreground=1.
        pred_line = 1.0 - pred_01
        target_line = 1.0 - target_01

        tau, T = 0.5, 0.1
        pred_line = torch.sigmoid((pred_line - tau) / T)
        target_line = torch.sigmoid((target_line - tau) / T)

        skel_pred = self.skel(pred_line)
        skel_target = self.skel(target_line)

        dims = (1, 2, 3)
        precision = (
            (skel_pred * target_line).sum(dims) + self.smooth
        ) / (skel_pred.sum(dims) + self.smooth)
        sensitivity = (
            (skel_target * pred_line).sum(dims) + self.smooth
        ) / (skel_target.sum(dims) + self.smooth)

        cldice = (
            2.0 * precision * sensitivity
            / (precision + sensitivity + 1e-8)
        )
        return 1.0 - cldice.mean()
