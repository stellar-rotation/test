"""Sobel 梯度损失 — 约束线条锐度，生成清晰边缘"""

import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    """在 Sobel 梯度域计算 L1 损失，迫使生成器关注边缘锐度"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.l1 = nn.L1Loss(reduction='none')

    def _gradient(self, x):
        """x: [B, C, H, W], 返回梯度幅值 [B, 1, H, W]"""
        if x.shape[1] > 1:
            g_mags = []
            for c in range(x.shape[1]):
                xc = x[:, c : c + 1]
                gx = nn.functional.conv2d(xc, self.sobel_x, padding=1)
                gy = nn.functional.conv2d(xc, self.sobel_y, padding=1)
                g_mags.append(torch.sqrt(gx**2 + gy**2 + 1e-6))
            return torch.mean(torch.stack(g_mags, dim=1), dim=1, keepdim=True)
        else:
            gx = nn.functional.conv2d(x, self.sobel_x, padding=1)
            gy = nn.functional.conv2d(x, self.sobel_y, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred, target, mask=None):
        grad_pred = self._gradient(pred)
        grad_target = self._gradient(target)
        loss = self.l1(grad_pred, grad_target)
        if mask is not None:
            return (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss.mean()
