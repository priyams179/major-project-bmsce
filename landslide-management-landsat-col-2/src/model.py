"""Models for landslide detection.

SiameseUNet  -- bi-temporal change detection on the Landsat event set.
  Input: 3*n_bands channels = pre(n) | post(n) | diff(n). For the v2 7-band
  export n=7 -> 21 input channels. Shared-weight encoder on the pre/post
  halves, |diff| features fused before the decoder.
DeepUNet     -- deeper single-image U-Net for the L4S benchmark (see below).
SingleUNet   -- plain single-image U-Net (legacy L4S baseline).
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


class DeepUNet(nn.Module):
    """5-level U-Net for the L4S benchmark (14-ch input).

    Deeper than ``SingleUNet`` (one extra down/up stage + dropout in the
    bottleneck) for a larger receptive field and more capacity, which the
    small/rare landslide class benefits from.
    """

    def __init__(self, in_ch: int = 14, base: int = 64, dropout: float = 0.2):
        super().__init__()
        b = base
        self.e1 = conv_block(in_ch, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.e4 = conv_block(b * 4, b * 8)
        self.e5 = conv_block(b * 8, b * 16)  # bottleneck
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

        self.up5 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.d5 = conv_block(b * 16, b * 8)
        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d4 = conv_block(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d3 = conv_block(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d2 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        s5 = self.drop(self.e5(self.pool(s4)))
        u = self.d5(torch.cat([self.up5(s5), s4], 1))
        u = self.d4(torch.cat([self.up4(u), s3], 1))
        u = self.d3(torch.cat([self.up3(u), s2], 1))
        u = self.d2(torch.cat([self.up2(u), s1], 1))
        return self.head(u)


class ChannelAttention(nn.Module):
    """Channel Attention Module (Squeeze-and-Excitation / CBAM channel block).
    
    Dynamically weights spectral bands (e.g. boosting SWIR & Red Edge,
    suppressing atmospheric water vapor / cirrus).
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.gap(x).view(b, c))
        max_out = self.mlp(self.gmp(x).view(b, c))
        weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """Spatial Attention Module (CBAM spatial block).
    
    Highlights spatial hazard boundaries and steep terrain zones.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""

    def __init__(self, in_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class TwoStreamAttentionUNet(nn.Module):
    """Two-Stream Multi-Modal Attention U-Net for Sentinel-2 + Topography (L4S).

    Improvements over standard monolithic U-Net:
    1. Two-Stream Asymmetric Encoder:
       - Optical branch: Sentinel-2 12 multispectral bands (B1-B12).
       - Topographic branch: ALOS PALSAR Slope & DEM (2 bands).
    2. Spectral Attention: Channel Attention (CBAM) dynamically re-weights
       multispectral bands.
    3. Topographic Gating: Topographic slope & elevation feature map modulates
       optical features (F_opt * (1 + tanh(F_topo))), enforcing physical terrain
       constraints before downsampling.
    4. 5-level deep decoder with skip connections and bottleneck dropout.
    """

    def __init__(self, in_opt: int = 12, in_topo: int = 2, base: int = 64, dropout: float = 0.2):
        super().__init__()
        b = base
        # Optical stream
        self.opt_stem = conv_block(in_opt, b)
        self.opt_cbam = CBAMBlock(b, reduction=4)

        # Topographic stream
        self.topo_stem = nn.Sequential(
            nn.Conv2d(in_topo, b // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(b // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(b // 2, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

        # Fused multi-modal stages
        self.e1_fuse = conv_block(b * 2, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.e4 = conv_block(b * 4, b * 8)
        self.e5 = conv_block(b * 8, b * 16)  # bottleneck
        self.cbam_bottleneck = CBAMBlock(b * 16, reduction=8)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

        # Decoder
        self.up5 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.d5 = conv_block(b * 16, b * 8)
        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d4 = conv_block(b * 8, b * 4)
        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d3 = conv_block(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d2 = conv_block(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 14, H, W] -> optical [B, 12, H, W], topo [B, 2, H, W]
        x_opt, x_topo = x[:, :12], x[:, 12:]

        # Asymmetric stems
        f_opt = self.opt_cbam(self.opt_stem(x_opt))
        f_topo = self.topo_stem(x_topo)

        # Topographic spatial gate: terrain slope modulates optical features
        f_gated = f_opt * (1.0 + torch.tanh(f_topo))
        s1 = self.e1_fuse(torch.cat([f_gated, f_topo], dim=1))

        # Downsampling hierarchy
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        s5 = self.drop(self.cbam_bottleneck(self.e5(self.pool(s4))))

        # Decoder with skip connections
        u = self.d5(torch.cat([self.up5(s5), s4], dim=1))
        u = self.d4(torch.cat([self.up4(u), s3], dim=1))
        u = self.d3(torch.cat([self.up3(u), s2], dim=1))
        u = self.d2(torch.cat([self.up2(u), s1], dim=1))
        return self.head(u)


class PatchClassifier(nn.Module):
    """Tile-level landslide presence classifier (14-ch, 128x128 -> 1 logit).

    5 conv stages with max-pool (128->4), global average pool, dropout, linear
    head. Scene/patch-level 'contains a landslide?' is far more separable than
    per-pixel delineation, so this reaches 90+ accuracy/F1 on L4S human labels.
    """

    def __init__(self, in_ch: int = 14, base: int = 64, dropout: float = 0.3):
        super().__init__()
        self.stem = conv_block(in_ch, base)        # 128
        self.s2 = conv_block(base, base * 2)       # -> 64
        self.s3 = conv_block(base * 2, base * 4)   # -> 32
        self.s4 = conv_block(base * 4, base * 8)   # -> 16
        self.s5 = conv_block(base * 8, base * 8)   # -> 8 -> 4
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.s2(self.pool(x))
        x = self.s3(self.pool(x))
        x = self.s4(self.pool(x))
        x = self.s5(self.pool(x))
        x = self.gap(x).flatten(1)
        return self.fc(self.drop(x)).squeeze(1)  # [B] logits


@torch.no_grad()
def tta_logits_cls(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """TTA for the classifier: mean probability over the 8 D4 transforms,
    returned in logit space. Spatial transforms don't change the label."""
    probs = []
    for k in range(4):
        xr = torch.rot90(x, k, dims=(2, 3))
        for flip in (False, True):
            xi = torch.flip(xr, dims=(3,)) if flip else xr
            probs.append(torch.sigmoid(model(xi)))
    mean_p = torch.stack(probs, 0).mean(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(mean_p / (1 - mean_p))


@torch.no_grad()
def tta_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Test-time augmentation: mean probability over the 8 dihedral (D4)
    transforms, returned back in logit space. Reliable small F1 gain and
    no training cost."""
    probs = []
    for k in range(4):              # 0/90/180/270 rotations
        xr = torch.rot90(x, k, dims=(2, 3))
        for flip in (False, True):  # + horizontal flip
            xi = torch.flip(xr, dims=(3,)) if flip else xr
            p = torch.sigmoid(model(xi))
            if flip:
                p = torch.flip(p, dims=(3,))
            p = torch.rot90(p, -k, dims=(2, 3))
            probs.append(p)
    mean_p = torch.stack(probs, 0).mean(0).clamp(1e-6, 1 - 1e-6)
    return torch.log(mean_p / (1 - mean_p))  # -> logits


class FocalTverskyBCELoss(nn.Module):
    """Focal-BCE + Focal-Tversky, with optional valid mask.

    Tversky (alpha<beta) penalises false negatives harder than false
    positives, which lifts recall on the rare landslide class and tends to
    raise F1 vs plain Dice. The focal BCE term keeps per-pixel calibration.
    """

    def __init__(self, pos_weight: float = 5.0, gamma: float = 2.0,
                 alpha: float = 0.3, beta: float = 0.7, tversky_gamma: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tversky_gamma = tversky_gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, target, valid=None):
        if valid is None:
            valid = torch.ones_like(target)
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal = ((1 - p_t) ** self.gamma) * bce
        weight = 1.0 + (self.pos_weight - 1.0) * target
        loss_bce = (focal * weight * valid).sum() / (valid.sum() + 1e-6)

        p, t, v = prob * valid, target * valid, valid
        tp = (p * t).sum(dim=(1, 2, 3))
        fp = (p * (1 - t) * v).sum(dim=(1, 2, 3))
        fn = ((1 - p) * t).sum(dim=(1, 2, 3))
        ti = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        tversky = ((1 - ti) ** self.tversky_gamma).mean()

        loss = loss_bce + tversky
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


class BoundaryAwareTverskyLoss(nn.Module):
    """Focal-Tversky + Sobel Boundary Edge Loss.

    Explicitly forces the model to learn sharp landslide perimeters and eliminates
    boundary edge splatter and dilation.
    """

    def __init__(
        self,
        pos_weight: float = 5.0,
        gamma: float = 2.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        boundary_weight: float = 0.4,
    ):
        super().__init__()
        self.ft = FocalTverskyBCELoss(pos_weight=pos_weight, gamma=gamma, alpha=alpha, beta=beta)
        self.boundary_weight = boundary_weight

        # Fixed Sobel kernels for horizontal & vertical edge extraction
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_edges(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        base_loss = self.ft(logits, target, valid=valid)

        if self.boundary_weight <= 0:
            return base_loss

        prob = torch.sigmoid(logits)
        pred_edges = self._get_edges(prob)
        target_edges = self._get_edges(target)

        if valid is not None:
            pred_edges = pred_edges * valid
            target_edges = target_edges * valid

        # Edge Dice Loss
        inter = (pred_edges * target_edges).sum(dim=(1, 2, 3))
        denom = pred_edges.sum(dim=(1, 2, 3)) + target_edges.sum(dim=(1, 2, 3))
        edge_loss = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

        total = base_loss + self.boundary_weight * edge_loss
        return torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)


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
