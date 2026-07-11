"""Masked Tversky loss for sparse black line foregrounds."""

import torch
import torch.nn as nn


class MaskedTverskyLoss(nn.Module):
    """Tversky loss evaluated only inside a binary damage mask."""

    def __init__(self, alpha=0.45, beta=0.55, smooth=1.0, temperature=0.1):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.temperature = temperature

    def forward(self, pred, target, damage_mask):
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got "
                f"{pred.shape} and {target.shape}"
            )
        if damage_mask.shape != pred.shape:
            raise ValueError(
                f"damage_mask must match pred shape, got "
                f"{damage_mask.shape} and {pred.shape}"
            )

        # Work in fp32 so reductions over 512x512 masks cannot overflow fp16.
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        mask = (damage_mask.float() > 0.5).float()

        # Line art is black (-1) on white (+1); convert black lines to foreground=1.
        pred_line = torch.sigmoid(-pred_fp32 / self.temperature)
        target_line = (1.0 - (target_fp32 + 1.0) * 0.5).clamp(0.0, 1.0)

        dims = tuple(range(1, pred.ndim))
        true_positive = (pred_line * target_line * mask).sum(dim=dims)
        false_positive = (pred_line * (1.0 - target_line) * mask).sum(dim=dims)
        false_negative = ((1.0 - pred_line) * target_line * mask).sum(dim=dims)

        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return 1.0 - score.mean()

