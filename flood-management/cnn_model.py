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
# Atrous Spatial Pyramid Pooling (ASPP)
# ---------------------------------------------------------------------------
class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Multi-dilation rate feature pooling for multi-scale water context."""

    def __init__(self, in_ch, out_ch, atrous_rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_ch, out_ch, rate))
        modules.append(ASPPPooling(in_ch, out_ch))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


# ---------------------------------------------------------------------------
# Pretrained ResNet-34 U-Net with ASPP Bottleneck
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Upsample via Bilinear + Conv, concatenate skip, and apply DoubleConv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        up_ch = in_ch // 2 if in_ch > 32 else in_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, up_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(up_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(up_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with Pretrained ResNet-34 Encoder and ASPP Multi-Scale Bottleneck.
    Supports arbitrary input channels (e.g. 2, 3, 4) with pretrained weight transfer.
    """

    def __init__(self, in_channels=2, pretrained=True):
        super().__init__()
        self.in_channels = in_channels
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = resnet34(weights=weights)
        except Exception:
            from torchvision.models import resnet34
            resnet = resnet34(pretrained=pretrained)

        # Input adaptation: modify first conv layer if in_channels != 3
        if in_channels == 3:
            self.conv1 = resnet.conv1
        else:
            orig_conv = resnet.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                if pretrained:
                    orig_w = orig_conv.weight
                    if in_channels < 3:
                        self.conv1.weight.copy_(orig_w[:, :in_channels, :, :])
                    else:
                        self.conv1.weight[:, :3, :, :].copy_(orig_w)
                        extra_w = orig_w.mean(dim=1, keepdim=True)
                        for c in range(3, in_channels):
                            self.conv1.weight[:, c:c+1, :, :].copy_(extra_w)
                else:
                    nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder stages
        self.layer1 = resnet.layer1  # 64 ch, stride 4
        self.layer2 = resnet.layer2  # 128 ch, stride 8
        self.layer3 = resnet.layer3  # 256 ch, stride 16
        self.layer4 = resnet.layer4  # 512 ch, stride 32

        # Bottleneck: ASPP
        self.bottleneck = ASPP(512, 256, atrous_rates=(6, 12, 18))

        # Decoder stages with skip connections
        self.up1 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=256)
        self.up2 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up3 = DecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
        self.up4 = DecoderBlock(in_ch=64, skip_ch=64, out_ch=32)

        # Final 2x upsample back to original image size
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(32, 32),
        )

        # Output head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x0))       # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)                     # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)                     # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                     # [B, 512, H/32, W/32]

        # Bottleneck
        b = self.bottleneck(x4)                  # [B, 256, H/32, W/32]

        # Decoder
        d1 = self.up1(b, x3)                     # [B, 256, H/16, W/16]
        d2 = self.up2(d1, x2)                    # [B, 128, H/8, W/8]
        d3 = self.up3(d2, x1)                    # [B, 64, H/4, W/4]
        d4 = self.up4(d3, x0)                    # [B, 32, H/2, W/2]

        out = self.final_up(d4)                  # [B, 32, H, W]
        return self.head(out)                    # [B, 1, H, W]


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------
def get_model(arch="unet", in_channels=2, pretrained=True, base_features=32):
    """
    Instantiate flood segmentation model by architecture name.
    """
    arch = arch.lower()
    if arch == "resnet34":
        return ResNet34UNet(in_channels=in_channels, pretrained=pretrained)
    elif arch == "unet":
        return UNet(in_channels=in_channels, base_features=base_features)
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: ['unet', 'resnet34']")


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
