"""
Segmentation models for landslide detection from multi-source satellite imagery.

Ported and extended from the flood-management pipeline (cnn_model.py).

Input configurations
--------------------
Config A – Full 14-channel stack:
    Sentinel-2 optical  : B2 (Blue), B3 (Green), B4 (Red), B8 (NIR),
                          B11 (SWIR-1), B12 (SWIR-2)          → 6 ch
    Sentinel-1 SAR      : VV, VH (σ⁰, log-scaled)             → 2 ch
    Terrain (DEM-derived): Elevation, Slope,
                           sin(Aspect), cos(Aspect)            → 4 ch
    Spectral indices    : dNDVI (pre-post difference),
                          NDWI (water index, post-event)       → 2 ch
    ─────────────────────────────────────────────────────────────
    Total                                                      = 14 ch

Config B – SAR-only monsoon (6-channel, cloud-robust):
    Sentinel-1 bi-temporal: VV_pre, VH_pre,
                            VV_post, VH_post                   → 4 ch
    Terrain               : Slope, TWI (Topographic Wetness)   → 2 ch
    ─────────────────────────────────────────────────────────────
    Total                                                      = 6 ch

Available models
----------------
UNet            – Lightweight plain U-Net; fast prototyping / edge deployment.
ResNet34UNet    – Pretrained ResNet-34 encoder + ASPP bottleneck + U-Net decoder.
get_model()     – Factory function that returns the requested architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# U-Net Building Blocks
# (Identical to the flood pipeline to allow weight-level comparison)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm2d → ReLU blocks.

    Parameters
    ----------
    in_ch  : int – Input channels.
    out_ch : int – Output channels.
    mid_ch : int or None – Intermediate channels (defaults to *out_ch*).
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.conv(x)


class Down(nn.Module):
    """MaxPool2d (stride-2) followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.poolconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.poolconv(x)


class Up(nn.Module):
    """
    ConvTranspose2d (stride-2) upsampling followed by DoubleConv with skip.

    Handles odd spatial sizes via centre-aligned zero-padding before the
    channel concatenation step.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 if spatial dimensions differ (e.g. odd input sizes)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# Atrous Spatial Pyramid Pooling (ASPP)
# ---------------------------------------------------------------------------

class ASPPConv(nn.Sequential):
    """Single dilated convolution branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    """Global average-pooling branch of ASPP (image-level context)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]          # remember spatial resolution
        for mod in self:
            x = mod(x)
        # Restore spatial resolution via bilinear interpolation
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale terrain / spectral context.

    Captures both local landslide texture and wider slope / drainage patterns
    by aggregating features at multiple dilation rates (6, 12, 18 by default).

    Parameters
    ----------
    in_ch        : int – Input feature channels (typically 512 from ResNet-34).
    out_ch       : int – Output channels for each branch before projection.
    atrous_rates : tuple – Dilation rates for the atrous convolutions.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        atrous_rates: tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()
        modules: list[nn.Module] = [
            # 1×1 conv branch
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_ch, out_ch, rate))
        modules.append(ASPPPooling(in_ch, out_ch))

        self.convs = nn.ModuleList(modules)
        # Project concatenated branches back to *out_ch*
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [conv(x) for conv in self.convs]
        return self.project(torch.cat(out, dim=1))


# ---------------------------------------------------------------------------
# Decoder Block (for ResNet34UNet)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Bilinear upsample + pointwise projection → cat skip connection → DoubleConv.

    Using bilinear upsampling (rather than transposed conv) reduces
    checkerboard artefacts in the final segmentation mask.

    Parameters
    ----------
    in_ch   : int – Channels from the lower-resolution feature map.
    skip_ch : int – Channels from the encoder skip connection.
    out_ch  : int – Output channels after DoubleConv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Halve channels during upsampling (unless already small)
        up_ch = in_ch // 2 if in_ch > 32 else in_ch
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, up_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(up_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(up_ch + skip_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Centre-align pad if spatial sizes differ
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# Plain U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Lightweight U-Net for binary landslide segmentation.

    Suitable for fast prototyping, ablation studies, or on-device deployment
    where a pretrained ImageNet backbone is not desired.

    Encoder depth: 32 → 64 → 128 → 256 → 512 feature maps (with base_features=32).
    Decoder mirrors the encoder with transposed-conv upsampling + skip connections.

    Parameters
    ----------
    in_channels   : int – Input channels (14 for Config A, 6 for Config B).
    base_features : int – Feature maps in the first encoder layer (default 32).

    Input configurations
    --------------------
    See module docstring for Config A (14-ch) and Config B (6-ch) details.
    """

    def __init__(self, in_channels: int = 14, base_features: int = 32):
        super().__init__()
        f = base_features

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16)

        # Decoder
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8,  f * 4)
        self.up3 = Up(f * 4,  f * 2)
        self.up4 = Up(f * 2,  f)

        # Output head (logit; apply sigmoid externally or inside loss)
        self.outc = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_channels, H, W]

        Returns
        -------
        Tensor [B, 1, H, W] – raw logits.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# ResNet-34 U-Net with ASPP Bottleneck
# ---------------------------------------------------------------------------

class ResNet34UNet(nn.Module):
    """
    U-Net decoder on top of a pretrained ResNet-34 encoder with ASPP bottleneck.

    N-channel Input Adaptation
    --------------------------
    ImageNet-pretrained ResNet-34 expects 3-channel (RGB) input, but our
    multi-source landslide stacks have 14 or 6 channels. The first
    convolutional layer is adapted as follows:

    * in_channels == 3  → reuse pretrained weights as-is.
    * in_channels  < 3  → copy only the first *in_channels* filters from the
                          pretrained 3-channel weight tensor (useful for greyscale
                          or 2-channel SAR-only variants).
    * in_channels  > 3  → copy all 3 pretrained filters into the first 3
                          channels of the new conv; initialise every additional
                          channel with the *channel-mean* of the pretrained
                          weights.  The mean initialisation is a well-established
                          heuristic (He et al., "Bag of Tricks") that gives the
                          new channels a neutral, well-scaled starting point while
                          preserving the ImageNet-learned spatial filters for
                          the original RGB channels.

    Parameters
    ----------
    in_channels : int  – Number of input channels (14 or 6).
    pretrained  : bool – Load ImageNet pretrained ResNet-34 weights (default True).

    Architecture
    ------------
    Encoder  : conv1 → layer1 (64) → layer2 (128) → layer3 (256) → layer4 (512)
    Bottleneck: ASPP(512 → 256) with dilation rates (6, 12, 18)
    Decoder  : 4 × DecoderBlock with skip connections from encoder stages
    Final up : 2× bilinear upsample + DoubleConv → 1×1 head
    """

    def __init__(self, in_channels: int = 14, pretrained: bool = True):
        super().__init__()
        self.in_channels = in_channels

        # ---- Load backbone -----------------------------------------------
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            resnet = resnet34(weights=weights)
        except Exception:
            # Fallback for older torchvision versions
            from torchvision.models import resnet34  # type: ignore[no-redef]
            resnet = resnet34(pretrained=pretrained)

        # ---- First conv: N-channel adaptation ----------------------------
        if in_channels == 3:
            # Reuse pretrained layer verbatim
            self.conv1 = resnet.conv1
        else:
            orig_conv = resnet.conv1          # pretrained [64, 3, 7, 7]
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                if pretrained:
                    orig_w = orig_conv.weight   # [64, 3, 7, 7]
                    if in_channels < 3:
                        # Fewer channels than RGB – copy first *in_channels* slices
                        self.conv1.weight.copy_(orig_w[:, :in_channels, :, :])
                    else:
                        # More channels than RGB – copy RGB filters, mean-init extras
                        self.conv1.weight[:, :3, :, :].copy_(orig_w)
                        ch_mean = orig_w.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                        for c in range(3, in_channels):
                            self.conv1.weight[:, c : c + 1, :, :].copy_(ch_mean)
                else:
                    nn.init.kaiming_normal_(
                        self.conv1.weight, mode="fan_out", nonlinearity="relu"
                    )

        # Remaining stem layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder stages (pretrained weights kept intact)
        self.layer1 = resnet.layer1   # [B,  64, H/4,  W/4 ]
        self.layer2 = resnet.layer2   # [B, 128, H/8,  W/8 ]
        self.layer3 = resnet.layer3   # [B, 256, H/16, W/16]
        self.layer4 = resnet.layer4   # [B, 512, H/32, W/32]

        # Bottleneck: multi-scale context from ASPP
        self.bottleneck = ASPP(512, 256, atrous_rates=(6, 12, 18))

        # Decoder: bilinear upsample + skip connections
        self.up1 = DecoderBlock(in_ch=256, skip_ch=256, out_ch=256)
        self.up2 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up3 = DecoderBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.up4 = DecoderBlock(in_ch=64,  skip_ch=64,  out_ch=32)

        # Final ×2 upsample back to input resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DoubleConv(32, 32),
        )

        # Binary output head
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_channels, H, W]

        Returns
        -------
        Tensor [B, 1, H, W] – raw logits.
        """
        # Encoder forward pass
        x0 = self.relu(self.bn1(self.conv1(x)))   # [B,  64, H/2,  W/2 ]
        x1 = self.layer1(self.maxpool(x0))         # [B,  64, H/4,  W/4 ]
        x2 = self.layer2(x1)                       # [B, 128, H/8,  W/8 ]
        x3 = self.layer3(x2)                       # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                       # [B, 512, H/32, W/32]

        # ASPP bottleneck
        b = self.bottleneck(x4)                    # [B, 256, H/32, W/32]

        # Decoder with skip connections
        d1 = self.up1(b,  x3)                      # [B, 256, H/16, W/16]
        d2 = self.up2(d1, x2)                      # [B, 128, H/8,  W/8 ]
        d3 = self.up3(d2, x1)                      # [B,  64, H/4,  W/4 ]
        d4 = self.up4(d3, x0)                      # [B,  32, H/2,  W/2 ]

        out = self.final_up(d4)                    # [B,  32, H,    W   ]
        return self.head(out)                      # [B,   1, H,    W   ]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_model(
    arch: str = "resnet34_unet",
    in_channels: int = 14,
    pretrained: bool = True,
    base_features: int = 32,
) -> nn.Module:
    """
    Instantiate a landslide segmentation model by architecture name.

    Parameters
    ----------
    arch          : str  – Architecture name: 'unet' or 'resnet34_unet'.
    in_channels   : int  – Input channels (14 for Config A, 6 for Config B).
    pretrained    : bool – Use ImageNet pretrained weights (ResNet34UNet only).
    base_features : int  – Base feature count for UNet (ignored for ResNet34UNet).

    Returns
    -------
    nn.Module ready for training.

    Raises
    ------
    ValueError – If *arch* is not one of the supported names.

    Examples
    --------
    >>> model = get_model('resnet34_unet', in_channels=14, pretrained=True)
    >>> model = get_model('unet', in_channels=6, base_features=32)
    """
    arch_lower = arch.lower()
    if arch_lower == "resnet34_unet":
        return ResNet34UNet(in_channels=in_channels, pretrained=pretrained)
    elif arch_lower == "unet":
        return UNet(in_channels=in_channels, base_features=base_features)
    else:
        raise ValueError(
            f"Unknown architecture '{arch}'. Supported: ['unet', 'resnet34_unet']"
        )
