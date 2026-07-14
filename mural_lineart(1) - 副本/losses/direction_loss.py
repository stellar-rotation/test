"""Line-direction consistency loss evaluated only near target edges."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionLoss(nn.Module):
    """Compare undirected gradient orientation around target lines."""

    def __init__(self, magnitude_threshold=0.1):
        super().__init__()
        self.magnitude_threshold = magnitude_threshold
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3))

    def _gradient(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        return gx, gy, magnitude

    def forward(self, pred, target):
        gx_pred, gy_pred, mag_pred = self._gradient(pred)
        gx_target, gy_target, mag_target = self._gradient(target)

        cosine = (
            gx_pred * gx_target + gy_pred * gy_target
        ) / (mag_pred * mag_target + 1e-6)
        orientation_error = 1.0 - cosine.abs().clamp(max=1.0)

        edge_mask = (mag_target > self.magnitude_threshold).to(pred.dtype)
        return (orientation_error * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
