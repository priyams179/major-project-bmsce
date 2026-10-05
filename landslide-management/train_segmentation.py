#!/usr/bin/env python3
"""
Phase 1: Train Multi-Modal Segmentation Model for Landslide Detection

Supports:
  - ResNet34-UNet with ASPP (default) and standard U-Net
  - Flexible N-channel inputs (6-ch SAR-only monsoon mode, 14-ch full multi-modal)
  - Masked Focal-Tversky loss for extreme class imbalance (<1% landslide pixels)
  - Landslide4Sense benchmark (3,799 patches, 128x128, 14 channels)
  - Custom India scene patches (256x256 or 512x512)
  - Mixed precision training (torch.amp)
  - Spatial block cross-validation (train/val split by India region)

Usage:
    python train_segmentation.py --arch resnet34_unet --in-channels 14 \\
        --data-dir data/processed/l4s --epochs 50 --batch-size 16
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
from torch.utils.data import DataLoader, Dataset

from models import (
    get_model,
    MaskedBCEDiceLoss,
    MaskedFocalTverskyLoss,
    ConfusionMatrixTracker,
)
from prepare_dataset import (
    S1_CHANNELS, LANDSAT_CHANNELS, DEM_CHANNELS, read_dataset_manifest,
    discover_split_samples, summarise_channels,
)


# The manifest helpers and split discovery live in prepare_dataset so that
# training, evaluation and inference cannot drift apart on either.
_discover_samples = discover_split_samples


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class LandslideDataset(Dataset):
    """
    Landslide segmentation dataset loader.

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
    """

    def __init__(self, data_dir: str, split: str = "train",
                 patch_size: int = 128, augment: bool = False,
                 l4s: bool = False):
        super().__init__()
        self.augment = augment
        self.patch_size = patch_size

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, valid_path = self.samples[idx]

        # --- Load image ---
        image = np.load(img_path).astype(np.float32)   # (H, W, C) or (C, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, ...]               # (1, H, W)
        elif image.ndim == 3 and image.shape[-1] < image.shape[0]:
            # (H, W, C) → (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        # --- Load mask ---
        mask = np.load(mask_path).astype(np.float32)    # (H, W) or (1, H, W)
        if mask.ndim == 3:
            mask = mask.squeeze(0)                       # → (H, W)

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

        # --- Augmentation ---
        if self.augment:
            image, mask, valid = self._augment(image, mask, valid)

        # --- Convert to tensors ---
        image_t = torch.from_numpy(image)                    # (C, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)
        valid_t = torch.from_numpy(valid).unsqueeze(0)       # (1, H, W)

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
        """Apply synchronised geometric augmentations to image, mask, and valid."""
        # RandomHorizontalFlip (p=0.5)
        if np.random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()
            valid = valid[:, ::-1].copy()

        # RandomVerticalFlip (p=0.5)
        if np.random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[::-1, :].copy()
            valid = valid[::-1, :].copy()

        # RandomRotate90 (k ∈ {0, 1, 2, 3})
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            valid = np.rot90(valid, k=k, axes=(0, 1)).copy()

        return image, mask, valid


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, scaler=None):
    """
    Evaluate model over a DataLoader using exact confusion-matrix tracking.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader  – yields (image, mask, valid_mask)
    criterion : nn.Module – masked loss
    device : torch.device
    scaler : optional GradScaler (used for AMP inference consistency; not needed
             for eval but kept for API symmetry)

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
def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-Modal Segmentation Model for Landslide Detection"
    )
    parser.add_argument(
        "--data-dir", default="data/processed/s1_landsat",
        help="Root data directory containing train/val/test sub-folders "
             "(default: data/processed/s1_landsat)"
    )
    parser.add_argument(
        "--arch", choices=["resnet34_unet", "unet"], default="resnet34_unet",
        help="Model architecture (default: resnet34_unet)"
    )
    parser.add_argument(
        "--in-channels", type=int, default=None,
        help=(
            "Input channels. Defaults to the count in the dataset manifest, "
            "which is authoritative: 7=S1 only, 16=S1+Landsat, "
            "24=S1+Landsat+DEM. Only pass this to override a wrong manifest."
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Initial learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--loss", choices=["focal_tversky", "bce_dice"], default="focal_tversky",
        help="Loss function (default: focal_tversky)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)"
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory to save checkpoints and artefacts (default: models/)"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable torch.amp automatic mixed precision (AMP)"
    )
    parser.add_argument(
        "--patch-size", type=int, default=128,
        help="Spatial patch size expected in dataset (default: 128)"
    )
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

    args.in_channels = in_channels

    print(f"Architecture    : {args.arch}")
    print(f"Input channels  : {in_channels}")
    if manifest["channel_names"]:
        groups = summarise_channels(manifest["channel_names"])
        print(f"Channel layout  : {groups}")
    print(f"Mixed precision : {use_amp}")

    # --- Datasets ---
    print("\nLoading dataset splits:")
    try:
        train_ds = LandslideDataset(
            args.data_dir, split="train",
            patch_size=args.patch_size, augment=True
        )
        val_ds = LandslideDataset(
            args.data_dir, split="val",
            patch_size=args.patch_size, augment=False
        )
        test_ds = LandslideDataset(
            args.data_dir, split="test",
            patch_size=args.patch_size, augment=False
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
    model = get_model(arch=args.arch, in_channels=args.in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.arch.upper()} (in_channels={args.in_channels}, "
          f"params={n_params:,})")

    # --- Loss ---
    if args.loss == "focal_tversky":
        criterion = MaskedFocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        print("Loss: Masked Focal Tversky Loss "
              "(alpha=0.3, beta=0.7, gamma=1.33 — optimised for high recall)")
    else:
        criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        print("Loss: Masked BCE + Dice Loss (50/50)")

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

        # Validate
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
    model = get_model(
        arch=best_ckpt.get("arch", args.arch),
        in_channels=best_ckpt.get("in_channels", args.in_channels)
    ).to(device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
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
