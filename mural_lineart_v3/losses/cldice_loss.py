"""
clDice Loss — 拓扑连续性约束，惩罚断线而非像素误差
参考: Shit et al., "clDice — a Novel Topology-Preserving Loss Function
      for Tubular Structure Segmentation", CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """可微软骨架化：形态学腐蚀/膨胀的 max/min-pool 近似"""

    def __init__(self, iterations=5):
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        s = x
        for _ in range(self.iterations):
            s = -F.max_pool2d(-s, kernel_size=3, stride=1, padding=1)
            s = F.max_pool2d(s, kernel_size=3, stride=1, padding=1)
        return s


class CLDiceLoss(nn.Module):
    """
    L_cldice = 1 - clDice(pred, target)

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens + eps)
      Tprec = |S_pred * V_target| / |S_pred|    骨架在目标体积内的比例
      Tsens = |S_target * V_pred| / |S_target|   目标骨架在预测体积内的比例

    配合 sigmoid 输出使用 (input_range='sigmoid')。
    """

    def __init__(self, iterations=5, sharpening_tau=0.5, sharpening_T=0.1):
        super().__init__()
        self.skeletonize = SoftSkeletonize(iterations)
        self.tau = sharpening_tau
        self.T = sharpening_T

    def _sharpen(self, x):
        return torch.sigmoid((x - self.tau) / self.T)

    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W] 或 [B, H, W]，值域 [0, 1]
        """
        # 强制FP32：512×512空间求和时FP16会溢出65504上限→inf→NaN
        pred = pred.float()
        target = target.float()

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        pred_s = self._sharpen(pred)
        target_s = self._sharpen(target)

        pred_skel = self.skeletonize(pred_s)
        target_skel = self.skeletonize(target_s)

        eps = 1e-6
        tprec = (pred_skel * target_s).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + eps)
        tsens = (target_skel * pred_s).sum(dim=(1, 2, 3)) / (target_skel.sum(dim=(1, 2, 3)) + eps)

        cldice = 2 * tprec * tsens / (tprec + tsens + eps)
        return (1 - cldice).mean()
