"""Direction Loss — 监督梯度方向（cos similarity），补 GradientLoss 的幅度盲区"""

import torch
import torch.nn as nn


class DirectionLoss(nn.Module):
    """
    在 Sobel 梯度域计算 1 - cos(theta)，惩罚线条方向错误。
    与 GradientLoss（监督梯度幅度）互补。

    L_dir = 1 - (g_pred · g_target) / (|g_pred| * |g_target| + eps)
    """

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

    def _gradient_vectors(self, x):
        """返回 (gx, gy)，各自 [B, C, H, W]"""
        gx = nn.functional.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        gy = nn.functional.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        return gx, gy

    def forward(self, pred, target, mask=None):
        """
        pred, target: [B, C, H, W]
        mask: [B, 1, H, W] or None
        """
        gx_p, gy_p = self._gradient_vectors(pred)
        gx_t, gy_t = self._gradient_vectors(target)

        eps = 1e-6
        dot = gx_p * gx_t + gy_p * gy_t
        norm_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + eps)
        norm_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + eps)
        cos_sim = dot / (norm_p * norm_t + eps)

        # 掩码：只关注线条区域（梯度显著的区域）
        if mask is None:
            # 自动生成：target 梯度显著处
            weight = (norm_t > 0.01).float().detach()
        else:
            weight = mask

        loss = (1.0 - cos_sim) * weight
        return loss.sum() / (weight.sum() + eps)
