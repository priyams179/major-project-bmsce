"""
SNUNet-CD: Siamese Nested U-Net for Bi-temporal Landslide Change Detection.

Architecture overview
---------------------
SNUNet-CD [Fang et al., 2021] is a change-detection network that combines
three ideas:

1. Siamese encoder – Two branches sharing identical ResNet-18 weights process
   the pre-event and post-event image independently.  The shared weights ensure
   that feature representations are comparable across time.

2. Nested dense skip connections (U-Net++ topology) – Instead of the single
   skip path in standard U-Net, features from every encoder depth are fused
   with all shallower decoder nodes via a dense connection pattern.  This
   recovers fine-grained spatial detail that is critical for delineating small
   landslide scarps and runout zones.

3. ECAM (Ensemble Channel Attention Module) – Multi-scale absolute-difference
   features (|post – pre|) are collected at several decoder depths, then
   adaptively weighted by a channel-attention mechanism before summation.  This
   lets the network emphasise the most discriminative bands for each scene.

Input convention
----------------
x_pre  : Tensor [B, in_channels, H, W] – pre-event patch
x_post : Tensor [B, in_channels, H, W] – post-event patch (same footprint)

Default in_channels = 5, meaning each temporal image contains:
    Red/NIR composite, SAR VV, SAR VH, DEM Slope, TWI

Output
------
Tensor [B, 1, H, W] of raw logits (apply torch.sigmoid to get probabilities).
A pixel value > 0.5 after sigmoid indicates a detected landslide change.

References
----------
Fang, S., Li, K., Shao, J., & Li, Z. (2022). SNUNet-CD: A densely connected
Siamese network for change detection of VHR images. IEEE Geoscience and Remote
Sensing Letters, 19, 1–5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive block
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """
    Conv2d → BatchNorm2d → ReLU (inplace).

    A thin wrapper used throughout SNUNet to keep the nested-skip
    connection blocks concise.

    Parameters
    ----------
    in_ch  : int – Input feature channels.
    out_ch : int – Output feature channels.
    kernel_size : int – Convolution kernel size (default 3, padding auto-set to preserve H×W).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2   # "same" padding for odd kernels
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# ECAM – Ensemble Channel Attention Module
# ---------------------------------------------------------------------------

class ECAM(nn.Module):
    """
    Ensemble Channel Attention Module (ECAM).

    Takes a *list* of feature maps at different scales (all expected to have
    the same spatial resolution after upsampling; the module handles any
    residual mismatch internally via bilinear interpolation).

    Mechanism
    ---------
    For each input feature map f_i of shape [B, C_i, H, W]:
      1. Global Average Pool → [B, C_i, 1, 1].
      2. Two FC layers (squeeze → excite) with intermediate ReLU and final
         Sigmoid to produce a channel weight vector w_i ∈ (0, 1)^{C_i}.
      3. Apply w_i element-wise to f_i.

    All weighted maps are then summed (after alignment) to produce a single
    attended output. The ensemble aggregation integrates context from multiple
    scales, which is important for detecting landslides of varying extent.

    Parameters
    ----------
    in_channels_list : list[int] – Channel counts for each input feature map.
    reduction        : int       – Squeeze ratio for the SE-like FC layers (default 4).
    out_channels     : int       – Channels of the final attended output.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        reduction: int = 4,
        out_channels: int = 64,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.attention_blocks = nn.ModuleList()

        for in_ch in in_channels_list:
            squeeze_ch = max(in_ch // reduction, 8)  # never go below 8 channels
            self.attention_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),              # [B, C, 1, 1]
                    nn.Flatten(),                         # [B, C]
                    nn.Linear(in_ch, squeeze_ch),
                    nn.ReLU(inplace=True),
                    nn.Linear(squeeze_ch, in_ch),
                    nn.Sigmoid(),
                )
            )

        # 1×1 conv to project each attended map to *out_channels* before sum
        self.proj_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
             for in_ch in in_channels_list]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        feature_list : list of Tensor [B, C_i, H_i, W_i]
            Feature maps at potentially different spatial resolutions; they
            will all be bilinearly resized to the size of the *first* tensor.

        Returns
        -------
        Tensor [B, out_channels, H_0, W_0] – attended ensemble output.
        """
        assert len(feature_list) == len(self.attention_blocks), (
            f"ECAM expected {len(self.attention_blocks)} tensors, "
            f"got {len(feature_list)}."
        )
        target_size = feature_list[0].shape[-2:]
        out = None

        for feat, attn_block, proj_conv in zip(
            feature_list, self.attention_blocks, self.proj_convs
        ):
            # ---- Channel attention ----------------------------------------
            w = attn_block(feat)                          # [B, C_i]
            w = w.unsqueeze(-1).unsqueeze(-1)             # [B, C_i, 1, 1]
            feat = feat * w                               # recalibrate channels

            # ---- Spatial alignment & projection ---------------------------
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            feat = proj_conv(feat)                        # [B, out_ch, H, W]

            out = feat if out is None else out + feat

        return self.relu(self.bn(out))


# ---------------------------------------------------------------------------
# Siamese ResNet-18 Encoder
# ---------------------------------------------------------------------------

def _build_siamese_encoder(in_channels: int, pretrained: bool) -> nn.ModuleDict:
    """
    Build a shared-weight ResNet-18 encoder with N-channel input adaptation.

    N-channel adaptation strategy (identical to ResNet34UNet in segmentation_model.py):
    - in_channels == 3 : reuse pretrained conv1 weights verbatim.
    - in_channels  < 3 : copy first *in_channels* slices of pretrained weights.
    - in_channels  > 3 : copy 3 pretrained filters; initialise extra channels
                         with the channel-mean of the pretrained weights.

    Returns a ModuleDict with keys:
        'conv1', 'bn1', 'relu', 'maxpool',
        'layer1', 'layer2', 'layer3', 'layer4'
    """
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18  # type: ignore[no-redef]
        resnet = resnet18(pretrained=pretrained)

    if in_channels == 3:
        conv1 = resnet.conv1
    else:
        orig_conv = resnet.conv1
        conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            if pretrained:
                orig_w = orig_conv.weight   # [64, 3, 7, 7]
                if in_channels < 3:
                    conv1.weight.copy_(orig_w[:, :in_channels, :, :])
                else:
                    conv1.weight[:, :3, :, :].copy_(orig_w)
                    ch_mean = orig_w.mean(dim=1, keepdim=True)
                    for c in range(3, in_channels):
                        conv1.weight[:, c : c + 1, :, :].copy_(ch_mean)
            else:
                nn.init.kaiming_normal_(
                    conv1.weight, mode="fan_out", nonlinearity="relu"
                )

    return nn.ModuleDict({
        "conv1":   conv1,
        "bn1":     resnet.bn1,
        "relu":    resnet.relu,
        "maxpool": resnet.maxpool,
        "layer1":  resnet.layer1,   # 64  ch, stride 4
        "layer2":  resnet.layer2,   # 128 ch, stride 8
        "layer3":  resnet.layer3,   # 256 ch, stride 16
        "layer4":  resnet.layer4,   # 512 ch, stride 32
    })


def _encode(encoder: nn.ModuleDict, x: torch.Tensor) -> list[torch.Tensor]:
    """
    Run a forward pass through the shared encoder and return multi-scale features.

    Returns
    -------
    [x0, x1, x2, x3, x4] where:
        x0 : [B,  64, H/2,  W/2 ]  – post-stem (before maxpool)
        x1 : [B,  64, H/4,  W/4 ]  – layer1 output
        x2 : [B, 128, H/8,  W/8 ]  – layer2 output
        x3 : [B, 256, H/16, W/16]  – layer3 output
        x4 : [B, 512, H/32, W/32]  – layer4 output
    """
    x0 = encoder["relu"](encoder["bn1"](encoder["conv1"](x)))  # [B, 64, H/2, W/2]
    x1 = encoder["layer1"](encoder["maxpool"](x0))             # [B,  64, H/4,  W/4]
    x2 = encoder["layer2"](x1)                                 # [B, 128, H/8,  W/8]
    x3 = encoder["layer3"](x2)                                 # [B, 256, H/16, W/16]
    x4 = encoder["layer4"](x3)                                 # [B, 512, H/32, W/32]
    return [x0, x1, x2, x3, x4]


# ---------------------------------------------------------------------------
# SNUNet-CD
# ---------------------------------------------------------------------------

class SNUNetCD(nn.Module):
    """
    Siamese Nested U-Net for bi-temporal landslide change detection (SNUNet-CD).

    The model processes a pre-event and a post-event image patch through a
    shared ResNet-18 encoder, computes absolute feature differences at each
    encoder scale, and decodes them through a nested U-Net++ decoder augmented
    with ECAM channel attention.

    Parameters
    ----------
    in_channels : int – Channels in each temporal image (default 5).
        Default 5-channel layout per temporal image:
            [Red/NIR composite, SAR VV, SAR VH, DEM Slope, TWI]
    num_classes : int – Output classes (default 1 → binary change map).
    pretrained  : bool – Initialise ResNet-18 backbone with ImageNet weights.
    base_ch     : int – Base channel count for decoder nodes (default 32).

    Input
    -----
    x_pre  : Tensor [B, in_channels, H, W]
    x_post : Tensor [B, in_channels, H, W]

    Output
    ------
    Tensor [B, num_classes, H, W] – raw logits.
    """

    # Channel counts from ResNet-18 encoder levels 0..4
    _ENC_CHS = [64, 64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 1,
        pretrained: bool = True,
        base_ch: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ---- Shared Siamese encoder (one set of weights, two forward passes) ----
        self.encoder = _build_siamese_encoder(in_channels, pretrained)

        # Encoder channel sizes for convenience
        ec = self._ENC_CHS   # [64, 64, 128, 256, 512]

        # ---- Dense nested decoder nodes -----------------------------------
        # Notation:  node_i_j = decoder node at scale i, density j
        # scale 0 = finest (stride-2 features, 64ch)
        # scale 4 = coarsest (stride-32 features, 512ch)
        #
        # Each node receives:
        #   - upsampled feature from the level below (after DoubleConv)
        #   - absolute difference feature from the Siamese encoder at this level
        #   - all previous decoder outputs at this scale (dense connections)

        # Scale 0 nodes  (from diff_0, 64ch encoder)
        # node_0_1 : diff_0 + up(diff_1)
        self.node_0_1 = ConvBnRelu(ec[0] + ec[1],        base_ch * 2)
        # node_0_2 : diff_0 + up(node_1_1) + node_0_1
        self.node_0_2 = ConvBnRelu(ec[0] + base_ch * 4 + base_ch * 2, base_ch * 2)
        # node_0_3 : diff_0 + up(node_1_2) + node_0_1 + node_0_2
        self.node_0_3 = ConvBnRelu(ec[0] + base_ch * 4 + base_ch * 2 * 2, base_ch * 2)

        # Scale 1 nodes  (128ch encoder)
        # node_1_1 : diff_1 + up(diff_2)
        self.node_1_1 = ConvBnRelu(ec[1] + ec[2],        base_ch * 4)
        # node_1_2 : diff_1 + up(node_2_1) + node_1_1
        self.node_1_2 = ConvBnRelu(ec[1] + base_ch * 8 + base_ch * 4, base_ch * 4)

        # Scale 2 nodes  (256ch encoder)
        # node_2_1 : diff_2 + up(diff_3)
        self.node_2_1 = ConvBnRelu(ec[2] + ec[3],        base_ch * 8)
        # node_2_2 : diff_2 + up(node_3_1) + node_2_1
        self.node_2_2 = ConvBnRelu(ec[2] + base_ch * 16 + base_ch * 8, base_ch * 8)

        # Scale 3 node   (512ch encoder)
        # node_3_1 : diff_3 + up(diff_4)
        self.node_3_1 = ConvBnRelu(ec[3] + ec[4],        base_ch * 16)

        # ---- ECAM: collect multi-scale difference outputs at scale 0 ------
        # We gather four outputs at the finest scale (node_0_1 through node_0_3
        # plus the raw diff_0) for ensemble channel attention.
        ecam_ch = base_ch * 2   # all scale-0 decoder nodes share this width
        self.ecam = ECAM(
            in_channels_list=[ec[0], base_ch * 2, base_ch * 2, base_ch * 2],
            reduction=4,
            out_channels=base_ch * 2,
        )

        # ---- Final upsample and output head --------------------------------
        # The ECAM output is at stride-2 resolution; upsample ×2 to full res.
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnRelu(base_ch * 2, base_ch * 2),
        )
        self.head = nn.Conv2d(base_ch * 2, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _upsample_add(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Bilinear upsample *x* to the spatial size of *skip*, then concatenate.
        """
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.cat([skip, x], dim=1)

    # ------------------------------------------------------------------
    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_pre  : Tensor [B, in_channels, H, W] – pre-event image patch.
        x_post : Tensor [B, in_channels, H, W] – co-registered post-event patch.

        Returns
        -------
        Tensor [B, num_classes, H, W] – raw change-detection logits.
        """
        # ---- Siamese encoding (shared weights) ---------------------------
        pre_feats  = _encode(self.encoder, x_pre)    # [f0, f1, f2, f3, f4]
        post_feats = _encode(self.encoder, x_post)   # [g0, g1, g2, g3, g4]

        # ---- Absolute difference features --------------------------------
        # Each diff_i captures the spectral/structural change at scale i.
        diff = [torch.abs(p - g) for p, g in zip(pre_feats, post_feats)]
        # diff[0]: [B,  64, H/2,  W/2 ]
        # diff[1]: [B,  64, H/4,  W/4 ]
        # diff[2]: [B, 128, H/8,  W/8 ]
        # diff[3]: [B, 256, H/16, W/16]
        # diff[4]: [B, 512, H/32, W/32]

        # ---- Nested decoder nodes (bottom-up, coarse to fine) -----------

        # Scale 3: coarsest difference expanded one level up
        n3_1 = self.node_3_1(self._upsample_add(diff[4], diff[3]))
        # → [B, base_ch*16, H/16, W/16]

        # Scale 2
        n2_1 = self.node_2_1(self._upsample_add(diff[3], diff[2]))
        # → [B, base_ch*8, H/8, W/8]
        n2_2 = self.node_2_2(
            torch.cat([diff[2],
                       F.interpolate(n3_1, size=diff[2].shape[-2:],
                                     mode="bilinear", align_corners=False),
                       n2_1], dim=1)
        )
        # → [B, base_ch*8, H/8, W/8]

        # Scale 1
        n1_1 = self.node_1_1(self._upsample_add(diff[2], diff[1]))
        # → [B, base_ch*4, H/4, W/4]
        n1_2 = self.node_1_2(
            torch.cat([diff[1],
                       F.interpolate(n2_2, size=diff[1].shape[-2:],
                                     mode="bilinear", align_corners=False),
                       n1_1], dim=1)
        )
        # → [B, base_ch*4, H/4, W/4]

        # Scale 0 (finest, stride-2)
        n0_1 = self.node_0_1(self._upsample_add(diff[1], diff[0]))
        # → [B, base_ch*2, H/2, W/2]

        n0_2 = self.node_0_2(
            torch.cat([diff[0],
                       F.interpolate(n1_2, size=diff[0].shape[-2:],
                                     mode="bilinear", align_corners=False),
                       n0_1], dim=1)
        )
        # → [B, base_ch*2, H/2, W/2]

        # For n0_3 we need a fresh n1_2-like feature; reuse n1_2 from above.
        n0_3 = self.node_0_3(
            torch.cat([diff[0],
                       F.interpolate(n1_2, size=diff[0].shape[-2:],
                                     mode="bilinear", align_corners=False),
                       n0_1, n0_2], dim=1)
        )
        # → [B, base_ch*2, H/2, W/2]

        # ---- ECAM ensemble attention at scale 0 --------------------------
        # Pool four scale-0 representations: raw diff + three decoder nodes.
        attended = self.ecam([diff[0], n0_1, n0_2, n0_3])
        # → [B, base_ch*2, H/2, W/2]

        # ---- Final upsample and prediction head --------------------------
        out = self.final_up(attended)   # [B, base_ch*2, H, W]
        return self.head(out)           # [B, num_classes, H, W]


# ---------------------------------------------------------------------------
# Change-detection model factory
# ---------------------------------------------------------------------------

def get_cd_model(
    arch: str = "snunet",
    in_channels: int = 5,
    num_classes: int = 1,
    pretrained: bool = True,
    base_ch: int = 32,
) -> nn.Module:
    """
    Instantiate a bi-temporal change-detection model by architecture name.

    Parameters
    ----------
    arch        : str  – Architecture name. Currently supported: 'snunet'.
    in_channels : int  – Channels per temporal image (default 5).
    num_classes : int  – Output classes (default 1 for binary change map).
    pretrained  : bool – Use ImageNet pretrained ResNet-18 backbone (default True).
    base_ch     : int  – Base channel width for decoder nodes (default 32).

    Returns
    -------
    nn.Module ready for training.

    Raises
    ------
    ValueError – If *arch* is not in the supported list.

    Example
    -------
    >>> cd_model = get_cd_model('snunet', in_channels=5, pretrained=True)
    >>> logits = cd_model(x_pre, x_post)   # → [B, 1, H, W]
    """
    arch_lower = arch.lower()
    if arch_lower == "snunet":
        return SNUNetCD(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            base_ch=base_ch,
        )
    else:
        raise ValueError(
            f"Unknown change-detection architecture '{arch}'. Supported: ['snunet']"
        )
