#!/usr/bin/env python3
"""
Landslide segmentation training with SOTA-levering additions:
  - winsorized per-split band stats (1-99 percentile) defending the per-split
    normalization from DEM/slope outliers;
  - synchronized MixUp over (image, mask) with soft mask targets so BCE/Dice
    losses see probabilistic positives;
  - --focal-tversky as a first-class loss choice (alpha/beta/gamma sweepable);
  - --tta flag: at inference, average hflip/vflip/rot90 predictions;
  - --optical-only flag: drop DEM/slope channels for the L4S benchmark
    paper's optical-only ablation column;
  - --focal-alpha/--focal-beta/--focal-gamma flags to sweep the Focal Tversky
    hyperparameters without touching code.

Usage:
    python train_segmentation.py --arch smp_deeplabv3plus_effb4 \\
        --loss bce_lovasz --norm split --scale 2 --mixup 0.2 --tta \\
        --data-dir data/processed/l4s --epochs 30 --batch-size 16 \\
        --output-dir models/m_sota_mixup_ft_tta

    # sweep focal tversky hyperparams
    python train_segmentation.py --loss focal_tversky --focal-alpha 0.25 \\
        --focal-beta 0.75 --focal-gamma 1.5 --norm split --scale 2 \\
        --data-dir data/processed/l4s --epochs 30 --batch-size 16 \\
        --output-dir models/m_ft_sweep_025_075_15
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import (
    get_model,
    CombinedLoss,
    MaskedBCELovaszLoss,
    MaskedBCEDiceLoss,
    MaskedFocalTverskyLoss,
    MaskedLovaszHingeLoss,
    ConfusionMatrixTracker,
)
from prepare_dataset import (
    S1_CHANNELS, LANDSAT_CHANNELS, DEM_CHANNELS, read_dataset_manifest,
    discover_split_samples, summarise_channels,
)


# The manifest helpers and split discovery live in prepare_dataset so that
# training, evaluation and inference cannot drift apart on either.
_discover_samples = discover_split_samples


def compute_or_load_split_stats(data_dir: str) -> dict:
    """
    Per-split, per-band mean/std over every patch, cached to split_stats.json.

    The L4S train sites differ from the val/test sites, and the band
    statistics differ accordingly (measured: aggregate std 0.85 train vs 1.26
    val).  The 2022 L4S first-place solution calls normalising each domain
    with its own statistics "separate normalization" and reports it as their
    single largest improvement.

    Before statistics are accumulated, each band is winsorized to its 1-99
    percentile per patch.  A handful of DEM / slope outliers (val band 9
    had raw std 3.4) would otherwise blow up the per-split std and defeat the
    normalization; the winsorization is the cheap first line of defense the
    L4S benchmark paper implicitly assumes when it relies on per-domain stats.
    """
    stats_path = os.path.join(data_dir, "split_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)

    stats: dict = {}
    for split in ("train", "val", "test"):
        samples = _discover_samples(os.path.join(data_dir, split), split)
        if not samples:
            continue
        s = sq = None
        n_px = 0
        for img_path, _, _ in samples:
            x = np.load(img_path).astype(np.float64)      # (C, H, W)
            if x.ndim == 2:
                x = x[np.newaxis]
            elif x.shape[-1] < x.shape[0]:
                x = np.transpose(x, (2, 0, 1))
            # Winsorize each band to its 1-99 percentile so a few extreme
            # DEM / slope outliers cannot blow up the per-split std (which
            # would otherwise defeat the per-split normalization).
            lo = np.percentile(x, 1.0, axis=(1, 2), keepdims=True)
            hi = np.percentile(x, 99.0, axis=(1, 2), keepdims=True)
            x = np.clip(x, lo, hi)
            if s is None:
                s = np.zeros(x.shape[0]); sq = np.zeros(x.shape[0])
            s += x.sum(axis=(1, 2)); sq += (x ** 2).sum(axis=(1, 2))
            n_px += x.shape[1] * x.shape[2]
        mean = s / n_px
        std = np.sqrt(np.maximum(sq / n_px - mean ** 2, 1e-12))
        stats[split] = {"mean": mean.tolist(), "std": std.tolist()}

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class LandslideDataset(Dataset):
    """
    Landslide segmentation dataset loader with synchronized MixUp.

    Supports two layouts:
      - Standard .npy layout: image_{i:05d}.npy, mask_{i:05d}.npy, valid_{i:05d}.npy
        inside <data_dir>/<split>/
      - Landslide4Sense layout (l4s=True): img/<split>/img{i:05d}.h5 etc.
        (currently via pre-converted .npy files produced by a preprocessing step)

    Parameters
    ----------
    data_dir : str
        Root directory containing split sub-folders.
    split : str
        'train', 'val', or 'test'.
    patch_size : int
        Spatial size of patches (expected, not enforced here).
    augment : bool
        Apply random geometric augmentations.
    l4s : bool
        If True, interpret directory layout as Landslide4Sense benchmark.
    norm : str
        Per-band normalization mode: 'none', 'train', 'split'.
    scale : int
        Scale-promotion factor (1 = no upscaling).
    stats : dict or None
        Per-split per-band mean/std for normalization.
    mixup : float
        MixUp alpha (0 = disabled). Applied only to training patches.
    optical_only : bool
        Drop the last two channels (DEM, slope) for the L4S benchmark
        paper's optical-only ablation column.
    """

    def __init__(self, data_dir: str, split: str = "train",
                 patch_size: int = 128, augment: bool = False,
                 l4s: bool = False, norm: str = "none",
                 scale: int = 1, stats: dict | None = None,
                 mixup: float = 0.0, optical_only: bool = False):
        super().__init__()
        self.augment = augment
        self.patch_size = patch_size
        if norm not in ("none", "train", "split"):
            raise ValueError(f"Unknown norm '{norm}' (use none|train|split)")
        self.norm = norm
        self.scale = int(scale)
        self.mixup = float(mixup)
        self.optical_only = bool(optical_only)
        # Resolve which per-band statistics apply to *this* split.
        self._mean = self._std = None
        if norm != "none":
            if not stats:
                raise ValueError(f"--norm {norm} requires split statistics")
            which = split if (norm == "split" and split in stats) else "train"
            self._mean = np.asarray(stats[which]["mean"], dtype=np.float32)
            self._std = np.asarray(stats[which]["std"], dtype=np.float32)

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: '{split_dir}'. "
                f"Expected structure: <data_dir>/train|val|test/"
            )

        # Discover samples by scanning for image_*.npy files.  prepare_dataset
        # writes split-prefixed names (train_image_00000.npy) while the older
        # L4S conversion wrote bare ones (image_00001.npy), so accept both.
        self.samples = _discover_samples(split_dir, split)
        if not self.samples:
            raise FileNotFoundError(
                f"No 'image_*.npy' files found in '{split_dir}'. "
                "Run prepare_dataset.py to generate patch files."
            )

        manifest = read_dataset_manifest(data_dir)
        self.channel_names = manifest["channel_names"]
        self.n_channels = manifest["n_channels"]

        if self.optical_only:
            if self._mean is not None:
                self._mean = self._mean[:-2].copy()
                self._std = self._std[:-2].copy()
            if self.n_channels is not None:
                self.n_channels -= 2
            if self.channel_names is not None:
                self.channel_names = self.channel_names[:-2]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, valid_path = self.samples[idx]

        # --- Load image ---
        image = np.load(img_path).astype(np.float32)   # (H, W, C) or (C, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, ...]               # (1, H, W)
        elif image.ndim == 3 and image.shape[-1] < image.shape[0]:
            # (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        # --- Optical-only ablation: drop DEM + slope channels ---
        if self.optical_only and image.shape[0] >= 2:
            image = image[:-2]

        # --- Load mask ---
        mask = np.load(mask_path).astype(np.float32)    # (H, W) or (1, H, W)
        if mask.ndim == 3:
            mask = mask.squeeze(0)                       # -> (H, W)

        # --- Load valid mask ---
        if os.path.exists(valid_path):
            valid = np.load(valid_path).astype(np.float32)
            if valid.ndim == 3:
                valid = valid.squeeze(0)
        else:
            valid = np.ones_like(mask)

        # Clamp to {0, 1}
        mask = np.clip(mask, 0.0, 1.0)
        valid = np.clip(valid, 0.0, 1.0)

        # --- Normalization (per-band) ---
        if self._mean is not None:
            image = (image - self._mean[:, None, None]) / (self._std[:, None, None] + 1e-6)

        # --- Augmentation (incl. MixUp) ---
        if self.augment:
            image, mask, valid = self._augment(image, mask, valid)

        # --- Convert to tensors ---
        image_t = torch.from_numpy(image.astype(np.float32))  # (C, H, W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        valid_t = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        # --- Scale promotion (L4S 2022 1st place: 128 px upsampling resists
        # the weak representation of small landslide branches) ---
        if self.scale > 1:
            image_t = F.interpolate(image_t.unsqueeze(0), scale_factor=self.scale,
                                    mode="bilinear", align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t.unsqueeze(0), scale_factor=self.scale,
                                   mode="nearest").squeeze(0)
            valid_t = F.interpolate(valid_t.unsqueeze(0), scale_factor=self.scale,
                                    mode="nearest").squeeze(0)

        # A stale directory can mix layouts across splits.  Catching it here
        # names the mismatch instead of surfacing as a conv1d shape error many
        # layers later.
        if self.n_channels is not None and image_t.shape[0] != self.n_channels:
            raise ValueError(
                f"Channel mismatch in {img_path}: file has "
                f"{image_t.shape[0]} channels, manifest declares "
                f"{self.n_channels}. Re-run prepare_dataset.py events."
            )

        return image_t, mask_t, valid_t

    def _augment(self, image: np.ndarray, mask: np.ndarray,
                 valid: np.ndarray):
        """Apply strong synchronized augmentations incl. MixUp to a training
        patch.  image is (C,H,W); mask/valid are (H,W).  Returns image (C,H,W),
        mask (H,W), valid (H,W)."""

        # --- MixUp over the image + soft mask (applied first, before geom aug) ---
        if self.mixup > 0:
            lam = np.random.beta(self.mixup, self.mixup)
            j = np.random.randint(0, len(self.samples))
            img_j, mask_j, _ = self._load_raw(j)
            if img_j.shape[0] != image.shape[0]:
                # safety: if channel counts somehow differ, skip mixup
                lam = 1.0
            image = lam * image + (1.0 - lam) * img_j
            # Soft mask target: linear blend keeps the BCE/Dice losses
            # well-behaved (they already expect probabilities in [0,1]).
            mask = lam * mask + (1.0 - lam) * mask_j
            # valid mask: keep a pixel valid only if both were valid.
            valid = valid * (valid > 0.5)

        # --- Geometric/strong augmentations (albumentations) ---
        img_hwc = np.transpose(image, (1, 2, 0)).astype(np.float32)  # (H,W,C)
        m_hwc = mask[..., np.newaxis].astype(np.float32)  # (H,W,1)
        v_hwc = valid[..., np.newaxis].astype(np.float32)  # (H,W,1)

        try:
            import albumentations as A
            # MixUp already makes the mask soft; GridDistortion can push mask
            # values slightly outside [0,1], so we clip after.
            aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(rotate=(-15, 15), translate_percent=(-0.05, 0.05),
                         scale=(0.9, 1.1), shear=(-5, 5), p=0.5),
                A.RandomResizedCrop(height=img_hwc.shape[0],
                                    width=img_hwc.shape[1],
                                    scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                    p=0.3),
                A.GaussNoise(var_limit=(1e-4, 5e-3), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1,
                                           contrast_limit=0.1, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.15),
            ], additional_targets={'mask': 'mask', 'valid': 'mask'})
            res = aug(image=img_hwc, mask=m_hwc, valid=v_hwc)
            img_hwc_a = res['image']
            m_hwc_a = res['mask']
            v_hwc_a = res['valid']
            image_a = np.transpose(img_hwc_a, (2, 0, 1))
            mask_a = np.clip(m_hwc_a[..., 0], 0.0, 1.0)   # keep soft mask in [0,1]
            valid_a = v_hwc_a[..., 0]
            return image_a, mask_a, valid_a
        except Exception:
            # fallback to original simple aug if albumentations fails
            if np.random.random() < 0.5:
                image = image[:, :, ::-1].copy(); mask = mask[:, ::-1].copy(); valid = valid[:, ::-1].copy()
            if np.random.random() < 0.5:
                image = image[:, ::-1, :].copy(); mask = mask[::-1, :].copy(); valid = valid[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy(); mask = np.rot90(mask, k=k, axes=(0, 1)).copy(); valid = np.rot90(valid, k=k, axes=(0, 1)).copy()
            return image, mask, valid

    def _load_raw(self, idx: int):
        """Load a raw (non-augmented, pre-MixUp) sample so MixUp can pair it."""
        img_path, mask_path, _ = self.samples[idx]
        image = np.load(img_path).astype(np.float32)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        elif image.ndim == 3 and image.shape[-1] < image.shape[0]:
            image = np.transpose(image, (2, 0, 1))
        if self.optical_only and image.shape[0] >= 2:
            image = image[:-2]
        mask = np.load(mask_path).astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = np.clip(mask, 0.0, 1.0)
        if self._mean is not None:
            image = (image - self._mean[:, None, None]) / (self._std[:, None, None] + 1e-6)
        return image, mask, None


# ---------------------------------------------------------------------------
# 2. Evaluation (with optional TTA)
# ---------------------------------------------------------------------------
def _tta_forward(model: nn.Module, x: torch.Tensor, device: torch.device,
                 scaler=None):
    """Average logits over identity + hflip + vflip + rot90(2) with proper inverse transforms."""
    def pred(t):
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                return model(t.to(device, non_blocking=True))
        return model(t.to(device, non_blocking=True))

    logits0 = pred(x)
    logits_h = torch.flip(pred(torch.flip(x, dims=[-1])), dims=[-1])
    logits_v = torch.flip(pred(torch.flip(x, dims=[-2])), dims=[-2])
    logits_rot = torch.rot90(pred(torch.rot90(x, k=2, dims=[-2, -1])), k=2, dims=[-2, -1])
    return (logits0 + logits_h + logits_v + logits_rot) / 4.0


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, scaler=None, tta: bool = False):
    """
    Evaluate model over a DataLoader using exact confusion-matrix tracking.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader  – yields (image, mask, valid_mask)
    criterion : nn.Module – masked loss
    device : torch.device
    scaler : optional GradScaler (kept for API symmetry; unused at eval)
    tta : bool – average identity/hflip/vflip/rot90(2) logits at inference

    Returns
    -------
    avg_loss : float
    metrics : dict  (iou, f1, precision, recall, accuracy, tp, fp, fn, tn)
    """
    model.eval()
    tracker = ConfusionMatrixTracker(threshold=0.5)
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if tta:
                logits = _tta_forward(model, x, device, scaler)
            else:
                logits = model(x)

            loss = criterion(logits, y, mask=mask)
            total_loss += loss.item()
            num_batches += 1
            tracker.update(logits, y, mask=mask)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = tracker.compute()
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def build_parser():
    """Module-level parser creator so --help and tests can import without
    triggering main()."""
    p = argparse.ArgumentParser(
        description="Train Multi-Modal Segmentation Model for Landslide Detection"
    )
    p.add_argument(
        "--data-dir", default="data/processed/s1_landsat",
        help="Root data directory containing train/val/test sub-folders "
             "(default: data/processed/s1_landsat)"
    )
    p.add_argument(
        "--arch", choices=["resnet34_unet", "unet", "smp_unet_effb4",
                           "smp_unet_mitb2", "smp_deeplabv3plus_effb4",
                           "smp_unet_resnet50"],
        default="resnet34_unet",
        help="Model architecture (default: resnet34_unet; smp_* = pretrained "
             "segmentation-models-pytorch encoders)"
    )
    p.add_argument(
        "--in-channels", type=int, default=None,
        help=(
            "Input channels. Defaults to the count in the dataset manifest, "
            "which is authoritative: 7=S1 only, 16=S1+Landsat, "
            "24=S1+Landsat+DEM. Only pass this to override a wrong manifest."
        ),
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    p.add_argument(
        "--lr", type=float, default=3e-4,
        help="Initial learning rate (default: 3e-4)"
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)"
    )
    p.add_argument(
        "--loss", choices=["focal_tversky", "bce_dice", "combined",
                           "lovasz", "bce_lovasz"],
        default="focal_tversky",
        help="Loss fn (default: focal_tversky). bce_lovasz = BCE + Lovasz hinge "
             "(L4S 2022 winners). Focal Tversky sweepable via --focal-alpha/--focal-beta/--focal-gamma."
    )
    p.add_argument(
        "--pos-weight", type=float, default=None,
        help=(
            "Positive-class weight for the BCE term. Landslides are about two "
            "percent of pixels in L4S (measured one to 42 neg:pos in train), "
            "so BCE alone is swamped; Dice compensates partly and this "
            "compensates the rest. Omit for unweighted."
        )
    )
    p.add_argument(
        "--base-features", type=int, default=32,
        help=(
            "Encoder width for --arch unet (default 32; 64 doubles the "
            "parameter count). Ignored by resnet34_unet."
        )
    )
    p.add_argument(
        "--decoder-dropout", type=float, default=0.0,
        help=(
            "Dropout probability in the resnet34_unet decoder blocks "
            "(default 0, i.e. off). The decoder is the only part trained from "
            "scratch, so that is where overfitting appears first."
        )
    )
    p.add_argument(
        "--norm", choices=["none", "train", "split"], default="none",
        help=(
            "Per-band input normalization. 'none' feeds raw values (previous "
            "behaviour). 'train' normalizes every split with train statistics; "
            "'split' normalizes each split with its own statistics ('separate "
            "normalization', the largest single gain in the L4S-2022 1st-place "
            "solution, targeting the train-vs-val/test distribution shift)."
        )
    )
    p.add_argument(
        "--scale", type=int, default=1,
        help=(
            "Scale promotion factor: bilinearly upsample every patch by this "
            "factor before the model (2 = 128->256 px). The L4S-2022 1st-place "
            "solution upscaled 128->512 to better represent small landslide "
            "branches. Higher values cost ~scale^2 compute and memory."
        )
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)"
    )
    p.add_argument(
        "--output-dir", default="models",
        help="Directory to save checkpoints and artefacts (default: models/)"
    )
    p.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable torch.amp automatic mixed precision (AMP)"
    )
    p.add_argument(
        "--patch-size", type=int, default=128,
        help="Spatial patch size expected in dataset (default: 128)"
    )
    p.add_argument(
        "--mixup", type=float, default=0.0,
        help="MixUp alpha over (image, soft_mask). 0 disables; ~0.1-0.3 is a "
             "strong regularizer when training from scratch on small RS datasets."
    )
    p.add_argument(
        "--tta", action="store_true",
        help="Use test-time augmentation at eval: average identity/hflip/vflip/"
             "rot90(2) logits. Slows eval ~4x; intended for final test report."
    )
    p.add_argument(
        "--optical-only", action="store_true",
        help="Drop the DEM/slope channels (last 2 of the 14-channel stack) so "
             "the model trains on optical-only input, matching the L4S "
             "benchmark paper's optical-only ablation column."
    )
    p.add_argument(
        "--focal-alpha", type=float, default=0.3,
        help="Focal Tversky alpha (lower values favour recall)."
    )
    p.add_argument(
        "--focal-beta", type=float, default=0.7,
        help="Focal Tversky beta (higher values favour recall)."
    )
    p.add_argument(
        "--focal-gamma", type=float, default=1.33,
        help="Focal Tversky gamma (sweep 1.0-2.0 to tune hard-example focus)."
    )
    return p


parser = build_parser()


def main():
    global parser
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.mixed_precision and device.type == "cuda"

    print("=" * 70)
    print("Landslide Detection — Multi-Modal Segmentation Training Pipeline")
    print("=" * 70)
    print(f"Device          : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {props.total_memory / 1e9:.2f} GB")
    # --- Channel count: manifest is authoritative -------------------------
    manifest = read_dataset_manifest(args.data_dir)
    if manifest["n_channels"] is None:
        if args.in_channels is None:
            print(
                f"[ERROR] No channel manifest under '{args.data_dir}' and no "
                f"--in-channels given.\n"
                f"        Re-run prepare_dataset.py events, or pass "
                f"--in-channels explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        in_channels = args.in_channels
        print(f"Input channels  : {in_channels} (no manifest; from CLI)")
    elif args.in_channels is not None and args.in_channels != manifest["n_channels"]:
        print(
            f"[WARNING] --in-channels {args.in_channels} disagrees with the "
            f"dataset manifest ({manifest['n_channels']} from "
            f"{manifest['source']}). Using the manifest."
        )
        in_channels = manifest["n_channels"]
    else:
        in_channels = manifest["n_channels"]

    # Optical-only: drop DEM (channel index -2) and slope (channel index -1)
    # from the 14-channel stack, matching the benchmark paper's optical-only
    # column.  We record the real input channel count separately so the model
    # is built with the post-dropout channel count.
    if args.optical_only:
        # The manifest's 14-channel stack ends with DEM + slope.
        in_channels = in_channels - 2
        print(f"Optical-only    : dropping DEM/slope channels -> "
              f"{in_channels} channels")

    args.in_channels = in_channels

    print(f"Architecture    : {args.arch}")
    print(f"Input channels  : {in_channels}")
    if manifest["channel_names"]:
        groups = summarise_channels(manifest["channel_names"])
        if args.optical_only:
            print(f"Channel layout  : {groups}  (DEM/slope dropped for optical-only)")
        else:
            print(f"Channel layout  : {groups}")
    print(f"Mixed precision : {use_amp}")
    print(f"MixUp alpha     : {args.mixup}")
    print(f"TTA at eval     : {args.tta}")
    print(f"Focal Tversky   : alpha={args.focal_alpha} beta={args.focal_beta} gamma={args.focal_gamma}")

    # --- Datasets ---
    print("\nLoading dataset splits:")
    split_stats = None
    if args.norm != "none":
        split_stats = compute_or_load_split_stats(args.data_dir)
        print(f"Normalization    : {args.norm} (per-band stats cached in "
              f"{os.path.join(args.data_dir, 'split_stats.json')})")
    if args.scale > 1:
        print(f"Scale promotion  : x{args.scale} "
              f"({args.patch_size}->{args.patch_size * args.scale} px)")
    try:
        train_ds = LandslideDataset(
            args.data_dir, split="train",
            patch_size=args.patch_size, augment=True,
            norm=args.norm, scale=args.scale, stats=split_stats,
            mixup=args.mixup, optical_only=args.optical_only,
        )
        val_ds = LandslideDataset(
            args.data_dir, split="val",
            patch_size=args.patch_size, augment=False,
            norm=args.norm, scale=args.scale, stats=split_stats,
            optical_only=args.optical_only,
        )
        test_ds = LandslideDataset(
            args.data_dir, split="test",
            patch_size=args.patch_size, augment=False,
            norm=args.norm, scale=args.scale, stats=split_stats,
            optical_only=args.optical_only,
        )
    except FileNotFoundError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

    if len(train_ds) == 0:
        print(f"[Error] No training patches found in '{args.data_dir}/train/'.",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Train : {len(train_ds):,} patches")
    print(f"  Val   : {len(val_ds):,} patches")
    print(f"  Test  : {len(test_ds):,} patches")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
        drop_last=(len(train_ds) > args.batch_size)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin
    )

    # --- Model ---
    model = get_model(
        arch=args.arch,
        in_channels=args.in_channels,
        base_features=args.base_features,
        decoder_dropout=args.decoder_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.arch.upper()} (in_channels={args.in_channels}, "
          f"base_features={args.base_features}, "
          f"decoder_dropout={args.decoder_dropout}, params={n_params:,})")

    # --- Loss ---
    if args.loss == "focal_tversky":
        criterion = MaskedFocalTverskyLoss(
            alpha=args.focal_alpha, beta=args.focal_beta, gamma=args.focal_gamma
        )
        print(f"Loss: Masked Focal Tversky Loss "
              f"(alpha={args.focal_alpha}, beta={args.focal_beta}, "
              f"gamma={args.focal_gamma} — optimised for high recall)")
    elif args.loss == "combined":
        criterion = CombinedLoss(ft_weight=0.5, dice_weight=0.5)
        print("Loss: Combined Focal Tversky + Dice (50/50)")
    elif args.loss == "lovasz":
        criterion = MaskedLovaszHingeLoss()
        print("Loss: Masked Lovasz-hinge (optimises IoU directly)")
    elif args.loss == "bce_lovasz":
        criterion = MaskedBCELovaszLoss(bce_weight=0.5, pos_weight=args.pos_weight)
        print("Loss: Masked BCE + Lovasz-hinge (50/50)")
    else:
        criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5,
                                      pos_weight=args.pos_weight)
        pw_txt = f", pos_weight={args.pos_weight}" if args.pos_weight else ""
        print(f"Loss: Masked BCE + Dice Loss (50/50{pw_txt})")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    # --- Training loop ---
    best_val_iou = -1.0
    history: dict = {
        "train_loss": [], "val_loss": [],
        "val_iou": [], "val_f1": [],
        "val_precision": [], "val_recall": [],
    }

    header = (
        f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
        f"{'Val IoU':>8} | {'Val F1':>8} | {'Precision':>9} | "
        f"{'Recall':>8} | {'Time':>6}"
    )
    print("\nStarting training:")
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0

        for x, y, mask in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = criterion(logits, y, mask=mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y, mask=mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss /= max(num_batches, 1)

        # Validate (no TTA during training — keep val cheap and stable)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, scaler)
        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | "
            f"{val_iou*100:>7.2f}% | {val_f1*100:>7.2f}% | "
            f"{val_prec*100:>8.2f}% | {val_rec*100:>7.2f}% | {elapsed:>5.1f}s"
        )

        # Save best checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "arch": args.arch,
                    "in_channels": args.in_channels,
                    # Recorded so inference can confirm it is fed the same
                    # channel layout this checkpoint was trained on.
                    "channel_names": manifest["channel_names"],
                    "optical_only": args.optical_only,
                    # Full config so any consumer can rebuild the exact model
                    # and loss this checkpoint was trained with.
                    "loss": args.loss,
                    "pos_weight": args.pos_weight,
                    "norm": args.norm,
                    "scale": args.scale,
                    "base_features": args.base_features,
                    "decoder_dropout": args.decoder_dropout,
                    "mixup": args.mixup,
                    "focal_alpha": args.focal_alpha,
                    "focal_beta": args.focal_beta,
                    "focal_gamma": args.focal_gamma,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                },
                best_path,
            )

    total_time = time.time() - t_start
    print("-" * len(header))
    print(
        f"Training complete in {total_time / 60:.1f} min. "
        f"Best Val IoU: {best_val_iou * 100:.2f}%"
    )

    # --- Final test evaluation ---
    print("\n" + "=" * 70)
    print("Evaluating Best Model on Test Split:")
    print("=" * 70)
    best_ckpt = torch.load(
        os.path.join(args.output_dir, "best_model.pth"),
        map_location=device, weights_only=False
    )
    # Rebuild exactly the model this checkpoint was trained with, including
    # the optical-only channel drop if it was used.
    ck_in_channels = best_ckpt.get("in_channels", args.in_channels)
    model = get_model(
        arch=best_ckpt.get("arch", args.arch),
        in_channels=ck_in_channels,
        base_features=best_ckpt.get("base_features", args.base_features),
        decoder_dropout=best_ckpt.get("decoder_dropout", args.decoder_dropout),
    ).to(device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device,
                                       scaler=scaler, tta=args.tta)
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Test IoU       : {test_metrics['iou'] * 100:.2f}%")
    print(f"  Test F1 / Dice : {test_metrics['f1'] * 100:.2f}%")
    print(f"  Test Precision : {test_metrics['precision'] * 100:.2f}%")
    print(f"  Test Recall    : {test_metrics['recall'] * 100:.2f}%")
    print(f"  Test Accuracy  : {test_metrics['accuracy'] * 100:.2f}%")
    print(
        f"  Pixel Counts   : "
        f"TP={test_metrics['tp']:,}, FP={test_metrics['fp']:,}, "
        f"FN={test_metrics['fn']:,}, TN={test_metrics['tn']:,}"
    )

    # --- Save history & plots ---
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_loss"], "b-o",
                 label="Train Loss", markersize=3)
        ax1.plot(epochs_range, history["val_loss"], "r-s",
                 label="Val Loss", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(
            epochs_range,
            [v * 100 for v in history["val_iou"]],
            "g-^", label="Val IoU (%)", markersize=3
        )
        ax2.plot(
            epochs_range,
            [v * 100 for v in history["val_f1"]],
            "m-d", label="Val F1 (%)", markersize=3
        )
        ax2.plot(
            epochs_range,
            [v * 100 for v in history["val_precision"]],
            "c-x", label="Val Precision (%)", markersize=3
        )
        ax2.plot(
            epochs_range,
            [v * 100 for v in history["val_recall"]],
            "y-p", label="Val Recall (%)", markersize=3
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score (%)")
        ax2.set_title("Validation Metrics")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {plot_path}")
    except Exception as exc:
        print(f"[Warning] Could not generate training plot: {exc}")

    print(f"Checkpoints saved to: '{args.output_dir}/'")


if __name__ == "__main__":
    main()
