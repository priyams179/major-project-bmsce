"""Loss functions matching the reference notebooks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Equal-weight BCE-with-logits + Dice loss.

    Works for one-hot multiclass targets and for the legacy segmentation
    pseudo-label path where the target is a batch of raw logits.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum(dim=(0, 2, 3))
        cardinality = probabilities.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth))
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


def dice_coefficient(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    target = target.float()
    intersection = (probabilities * target).sum(dim=(0, 2, 3))
    cardinality = probabilities.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return ((2.0 * intersection + smooth) / (cardinality + smooth)).mean()


def dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    return 1.0 - dice_coefficient(probabilities, target, smooth)
