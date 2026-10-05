#!/usr/bin/env python3
"""
Phase 2: Train Bi-Temporal Change Detection Model for Landslide Mapping

Trains SNUNet-CD on pre/post event image pairs to detect new landslides.
Critical for India monsoon operations: SAR pre/post pairs available within
12 days (Sentinel-1 revisit), while optical may take 14-85 days due to clouds.

Input pairs: (pre_image, post_image) where each is (B, C, H, W)
Output: Binary change map — 1=new landslide, 0=no change

Usage:
    python train_change_detection.py --arch snunet --in-channels 5 \\
        --data-dir data/processed/bitemporal --epochs 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import (
    get_cd_model,
    MaskedBCEDiceLoss,
    MaskedFocalTverskyLoss,
    ConfusionMatrixTracker,
)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class BiTemporalDataset(Dataset):
    """
    Bi-temporal change detection dataset for landslide mapping.

    Expects patch triplets inside <data_dir>/<split>/:
        pre_{i}.npy          – pre-event image  (H, W, C) or (C, H, W)
        post_{i}.npy         – post-event image (H, W, C) or (C, H, W)
        change_mask_{i}.npy  – binary change label (H, W) or (1, H, W)
        valid_{i}.npy        – valid pixel mask   (H, W) or (1, H, W) [optional]

    Augmentations are applied **synchronously** to both pre and post images,
    the change mask, and the valid mask so spatial correspondence is preserved.

    Parameters
    ----------
    data_dir : str
        Root directory with split sub-folders.
    split : str
        'train', 'val', or 'test'.
    augment : bool
        Apply random geometric augmentations during training.
    """

    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = False):
        super().__init__()
        self.augment = augment

        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Split directory not found: '{split_dir}'. "
                f"Expected structure: <data_dir>/train|val|test/"
            )

        import glob
        pre_files = sorted(glob.glob(os.path.join(split_dir, "pre_*.npy")))
        if not pre_files:
            raise FileNotFoundError(
                f"No 'pre_*.npy' files found in '{split_dir}'. "
                "Run preprocessing to generate bi-temporal patch files."
            )

        self.samples = []
        for pre_path in pre_files:
            basename = os.path.basename(pre_path)          # pre_0.npy
            idx_str = basename.replace("pre_", "").replace(".npy", "")  # 0
            post_path = os.path.join(split_dir, f"post_{idx_str}.npy")
            change_path = os.path.join(split_dir, f"change_mask_{idx_str}.npy")
            valid_path = os.path.join(split_dir, f"valid_{idx_str}.npy")

            if not os.path.exists(post_path):
                raise FileNotFoundError(
                    f"Missing post-event file: '{post_path}'"
                )
            if not os.path.exists(change_path):
                raise FileNotFoundError(
                    f"Missing change mask file: '{change_path}'"
                )
            self.samples.append((pre_path, post_path, change_path, valid_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"BiTemporalDataset is empty for split='{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pre_path, post_path, change_path, valid_path = self.samples[idx]

        pre = self._load_image(pre_path)
        post = self._load_image(post_path)

        change = np.load(change_path).astype(np.float32)
        if change.ndim == 3:
            change = change.squeeze(0)
        change = np.clip(change, 0.0, 1.0)

        if os.path.exists(valid_path):
            valid = np.load(valid_path).astype(np.float32)
            if valid.ndim == 3:
                valid = valid.squeeze(0)
        else:
            valid = np.ones_like(change)
        valid = np.clip(valid, 0.0, 1.0)

        if self.augment:
            pre, post, change, valid = self._augment(pre, post, change, valid)

        pre_t = torch.from_numpy(pre)
        post_t = torch.from_numpy(post)
        change_t = torch.from_numpy(change).unsqueeze(0)   # (1, H, W)
        valid_t = torch.from_numpy(valid).unsqueeze(0)     # (1, H, W)

        return pre_t, post_t, change_t, valid_t

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load .npy image and ensure (C, H, W) float32 format."""
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
            # (H, W, C) → (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _augment(self, pre: np.ndarray, post: np.ndarray,
                 change: np.ndarray, valid: np.ndarray):
        """Apply identical geometric augmentations to all arrays."""
        # RandomHorizontalFlip (p=0.5)
        if np.random.random() < 0.5:
            pre = pre[:, :, ::-1].copy()
            post = post[:, :, ::-1].copy()
            change = change[:, ::-1].copy()
            valid = valid[:, ::-1].copy()

        # RandomVerticalFlip (p=0.5)
        if np.random.random() < 0.5:
            pre = pre[:, ::-1, :].copy()
            post = post[:, ::-1, :].copy()
            change = change[::-1, :].copy()
            valid = valid[::-1, :].copy()

        # RandomRotate90 (k ∈ {0, 1, 2, 3})
        k = np.random.randint(0, 4)
        if k > 0:
            pre = np.rot90(pre, k=k, axes=(1, 2)).copy()
            post = np.rot90(post, k=k, axes=(1, 2)).copy()
            change = np.rot90(change, k=k, axes=(0, 1)).copy()
            valid = np.rot90(valid, k=k, axes=(0, 1)).copy()

        return pre, post, change, valid


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------
def evaluate_cd(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                device: torch.device):
    """
    Evaluate a change detection model using exact confusion-matrix tracking.

    Parameters
    ----------
    model : nn.Module  – accepts (pre, post) tensor pair
    loader : DataLoader – yields (pre, post, change_mask, valid_mask)
    criterion : nn.Module – masked loss
    device : torch.device

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
        for pre, post, change, valid in loader:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            change = change.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            logits = model(pre, post)
            loss = criterion(logits, change, mask=valid)

            total_loss += loss.item()
            num_batches += 1
            tracker.update(logits, change, mask=valid)

    avg_loss = total_loss / max(num_batches, 1)
    metrics = tracker.compute()
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Bi-Temporal Change Detection Model for Landslide Mapping"
    )
    parser.add_argument(
        "--data-dir", default="data/processed/bitemporal",
        help="Root data directory with train/val/test sub-folders "
             "(default: data/processed/bitemporal)"
    )
    parser.add_argument(
        "--arch", choices=["snunet", "siamese_unet"], default="snunet",
        help="Change detection architecture (default: snunet)"
    )
    parser.add_argument(
        "--in-channels", type=int, default=7,
        help=(
            "Channels per temporal image. The active pipeline feeds each date "
            "the full S1 bitemporal stack, so 7 = "
            "vv_pre, vh_pre, vv_post, vh_post, delta_vv, delta_vh, vh_vv_ratio "
            "(default: 7). Legacy SAR-only sets use 5."
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.mixed_precision and device.type == "cuda"

    print("=" * 70)
    print("Landslide Mapping — Bi-Temporal Change Detection Training Pipeline")
    print("=" * 70)
    print(f"Device          : {device}")
    if device.type == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Architecture    : {args.arch}")
    print(f"Input channels  : {args.in_channels} per temporal image")
    print(f"Mixed precision : {use_amp}")

    # --- Datasets ---
    print("\nLoading dataset splits:")
    try:
        train_ds = BiTemporalDataset(args.data_dir, split="train", augment=True)
        val_ds = BiTemporalDataset(args.data_dir, split="val", augment=False)
        test_ds = BiTemporalDataset(args.data_dir, split="test", augment=False)
    except FileNotFoundError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

    if len(train_ds) == 0:
        print(
            f"[Error] No training patches found in '{args.data_dir}/train/'.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"  Train : {len(train_ds):,} patch pairs")
    print(f"  Val   : {len(val_ds):,} patch pairs")
    print(f"  Test  : {len(test_ds):,} patch pairs")

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
    model = get_cd_model(arch=args.arch, in_channels=args.in_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.arch.upper()} "
          f"(in_channels={args.in_channels}×2, params={n_params:,})")

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

        for pre, post, change, valid in train_loader:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            change = change.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(pre, post)
                    loss = criterion(logits, change, mask=valid)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(pre, post)
                loss = criterion(logits, change, mask=valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss /= max(num_batches, 1)

        # Validate
        val_loss, val_metrics = evaluate_cd(model, val_loader, criterion, device)
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
            best_path = os.path.join(args.output_dir, "best_cd_model.pth")
            torch.save(
                {
                    "arch": args.arch,
                    "in_channels": args.in_channels,
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
    print("Evaluating Best CD Model on Test Split:")
    print("=" * 70)
    best_ckpt = torch.load(
        os.path.join(args.output_dir, "best_cd_model.pth"),
        map_location=device, weights_only=False
    )
    model = get_cd_model(
        arch=best_ckpt.get("arch", args.arch),
        in_channels=best_ckpt.get("in_channels", args.in_channels)
    ).to(device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_metrics = evaluate_cd(model, test_loader, criterion, device)
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
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "final_cd_model.pth"))
    history_path = os.path.join(args.output_dir, "cd_training_history.json")
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
        ax1.set_title("Training & Validation Loss (Change Detection)")
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
        ax2.set_title("Validation Metrics (Change Detection)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "cd_training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {plot_path}")
    except Exception as exc:
        print(f"[Warning] Could not generate training plot: {exc}")

    print(f"Checkpoints saved to: '{args.output_dir}/'")


if __name__ == "__main__":
    main()
