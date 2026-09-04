"""
U-Net architecture and masked loss functions for flood detection from SAR imagery.

Supports:
- Flexible input channels (2-channel [VV, VH] or 3-channel [VV, VH, difference])
- Masked BCE + Dice loss (ignores invalid / nodata / cloud pixels where target == -1)
- Masked Focal Tversky loss (for extreme class imbalance)
- Exact dataset-level Confusion Matrix metrics (IoU, F1, Precision, Recall)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net Building Blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.poolconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.poolconv(x)


class Up(nn.Module):
    """Upscale then DoubleConv with skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if spatial sizes differ
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary flood segmentation from SAR imagery.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 2 for VV+VH, 3 for VV+VH+diff).
    base_features : int
        Feature maps in the initial layer (default 32).
    """

    def __init__(self, in_channels=2, base_features=32):
        super().__init__()
        f = base_features

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)

        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)

        self.outc = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Masked Loss Functions
# ---------------------------------------------------------------------------
class MaskedDiceLoss(nn.Module):
    """Soft Dice loss computed strictly over valid mask pixels."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)

        if mask is not None:
            probs = probs * mask
            targets = targets * mask

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1)
            valid_idx = mask_flat > 0.5
            if not valid_idx.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            probs_flat = probs_flat[valid_idx]
            targets_flat = targets_flat[valid_idx]

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class MaskedBCEDiceLoss(nn.Module):
    """
    Combined Masked Binary Cross Entropy + Masked Soft Dice Loss.
    Correctly ignores nodata / invalid pixels (where mask == 0).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.dice = MaskedDiceLoss()

    def forward(self, logits, targets, mask=None):
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        bce_loss_raw = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )

        if mask is not None:
            valid_pixels = mask.sum()
            if valid_pixels > 0:
                bce_loss = (bce_loss_raw * mask).sum() / valid_pixels
            else:
                bce_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            bce_loss = bce_loss_raw.mean()

        dice_loss = self.dice(logits, targets, mask=mask)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MaskedFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss on valid pixels.
    Optimizes precision (alpha) and recall (beta) weighting for imbalanced segmentation.
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # weight on false positives (higher -> higher precision)
        self.beta = beta    # weight on false negatives (higher -> higher recall)
        self.gamma = gamma  # focal parameter
        self.smooth = smooth

    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)

        if mask is not None:
            probs = probs * mask
            targets = targets * mask

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1)
            valid = mask_flat > 0.5
            if not valid.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            probs_flat = probs_flat[valid]
            targets_flat = targets_flat[valid]

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)
        return focal_tversky


# ---------------------------------------------------------------------------
# Confusion Matrix & Metric Tracker
# ---------------------------------------------------------------------------
class ConfusionMatrixTracker:
    """
    Tracks True Positives, False Positives, False Negatives, True Negatives
    across batches to compute exact dataset-level classification metrics.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits, targets, mask=None):
        preds = (torch.sigmoid(logits) > self.threshold).bool()
        targets = (targets > 0.5).bool()

        if mask is not None:
            valid = (mask > 0.5).bool()
            preds = preds[valid]
            targets = targets[valid]
        else:
            preds = preds.view(-1)
            targets = targets.view(-1)

        self.tp += int((preds & targets).sum().item())
        self.fp += int((preds & (~targets)).sum().item())
        self.fn += int(((~preds) & targets).sum().item())
        self.tn += int(((~preds) & (~targets)).sum().item())

    def compute(self) -> dict:
        total_pos = self.tp + self.fn
        total_pred_pos = self.tp + self.fp
        total_valid = self.tp + self.fp + self.fn + self.tn

        precision = self.tp / total_pred_pos if total_pred_pos > 0 else 0.0
        recall = self.tp / total_pos if total_pos > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        union = self.tp + self.fp + self.fn
        iou = self.tp / union if union > 0 else 0.0
        accuracy = (self.tp + self.tn) / total_valid if total_valid > 0 else 0.0

        return {
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


# Compatibility alias
BCEDiceLoss = MaskedBCEDiceLoss
