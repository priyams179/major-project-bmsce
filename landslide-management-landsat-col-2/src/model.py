"""Siamese U-Net for bi-temporal landslide change detection.

Input: 9 channels = pre(3) | post(3) | diff(3)
Two-stream design: shared conv encoder on pre/post halves, difference
features fused before decoder. Lightweight for GPU training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Shared-weight encoder: cin channels -> multi-scale features."""

    def __init__(self, cin: int = 3, base: int = 32):
        super().__init__()
        self.b = base
        self.e1 = conv_block(cin, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.e4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        d1 = self.e1(x)          # [B, C, H, W]
        d2 = self.e2(self.pool(d1))
        d3 = self.e3(self.pool(d2))
        d4 = self.e4(self.pool(d3))
        return d1, d2, d3, d4


class SiameseUNet(nn.Module):
    """Siamese encoder (shared weights) + difference fusion + U-Net decoder."""

    def __init__(self, in_ch: int = 9, base: int = 32):
        super().__init__()
        assert in_ch % 3 == 0, "in_ch must be 3 * n_bands (pre|post|diff)"
        self.n_bands = in_ch // 3
        self.enc = Encoder(self.n_bands, base)

        # Fuse pre/post/|diff| features: input 3*C -> C at each scale.
        self.fuse = nn.ModuleList(
            [conv_block(base * (2**i) * 3, base * (2**i)) for i in range(4)]
        )

        b = base
        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d4 = conv_block(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d3 = conv_block(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d2 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3*n_bands, H, W] -> pre, post, diff each n_bands ch
        n = self.n_bands
        pre, post = x[:, :n], x[:, n : 2 * n]

        p1, p2, p3, p4 = self.enc(pre)
        q1, q2, q3, q4 = self.enc(post)

        f1 = self.fuse[0](torch.cat([p1, q1, (p1 - q1).abs()], 1))
        f2 = self.fuse[1](torch.cat([p2, q2, (p2 - q2).abs()], 1))
        f3 = self.fuse[2](torch.cat([p3, q3, (p3 - q3).abs()], 1))
        f4 = self.fuse[3](torch.cat([p4, q4, (p4 - q4).abs()], 1))

        u = self.d4(torch.cat([self.up4(f4), f3], 1))
        u = self.d3(torch.cat([self.up3(u), f2], 1))
        u = self.d2(torch.cat([self.up2(u), f1], 1))
        return self.head(u)  # [B,1,H,W] logits


class SingleUNet(nn.Module):
    """Plain single-image U-Net (L4S benchmark): encoder + decoder, in_ch input."""

    def __init__(self, in_ch: int = 14, base: int = 32):
        super().__init__()
        self.enc = Encoder(in_ch, base)
        b = base
        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d4 = conv_block(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d3 = conv_block(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d2 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, d2, d3, d4 = self.enc(x)
        u = self.d4(torch.cat([self.up4(d4), d3], 1))
        u = self.d3(torch.cat([self.up3(u), d2], 1))
        u = self.d2(torch.cat([self.up2(u), d1], 1))
        return self.head(u)


class DiceBCELoss(nn.Module):
    """Focal BCE + Dice. Ignores invalid pixels via valid mask.

    Focal term down-weights easy background (vast majority) so the
    model focuses on hard landslide pixels.
    """

    def __init__(self, pos_weight: float = 5.0, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, valid=None):
        if valid is None:
            valid = torch.ones_like(target)
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        # Focal modulation.
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = ((1 - p_t) ** self.gamma) * bce
        weight = 1.0 + (self.pos_weight - 1.0) * target
        loss_bce = (focal * weight * valid).sum() / (valid.sum() + 1e-6)

        inter = (prob * target * valid).sum(dim=(1, 2, 3))
        denom = (prob * valid).sum(dim=(1, 2, 3)) + (target * valid).sum(dim=(1, 2, 3))
        dice = 1.0 - (2 * inter + 1) / (denom + 1)
        loss = loss_bce + dice.mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


@torch.no_grad()
def metrics(logits, target, valid=None, thr: float = 0.5) -> dict:
    """Precision, recall, F1, IoU on positive class (valid pixels only)."""
    if valid is None:
        valid = torch.ones_like(target)
    pred = (torch.sigmoid(logits) > thr).float()
    v = valid
    tp = ((pred == 1) & (target == 1) & (v == 1)).sum().item()
    fp = ((pred == 1) & (target == 0) & (v == 1)).sum().item()
    fn = ((pred == 0) & (target == 1) & (v == 1)).sum().item()
    tn = ((pred == 0) & (target == 0) & (v == 1)).sum().item()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "iou": iou, "acc": acc}
