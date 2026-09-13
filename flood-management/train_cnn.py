#!/usr/bin/env python3
"""
Train Advanced Deep Learning Models on Sen1Floods11 & Multi-Dataset SAR Imagery.

Features:
  - Supports Pretrained ResNet-34 U-Net with ASPP multi-scale bottleneck & standard U-Net
  - Flexible 2-channel [VV, VH] or 3-channel [VV, VH, (VH - VV) difference ratio] inputs
  - Scalable to multi-dataset training (including 814 global permanent water chips)
  - Masked Focal-Tversky loss (for high recall) and Masked BCE + Dice loss
  - Real-time geometric data augmentations (flips, 90-degree rotations)
  - Exact dataset-level Confusion Matrix tracking (IoU, F1, Precision, Recall, Accuracy)

Usage:
    uv run python train_cnn.py --arch resnet34 --in-channels 3 --loss focal_tversky --epochs 25
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn_model import (
    get_model,
    MaskedBCEDiceLoss,
    MaskedFocalTverskyLoss,
    ConfusionMatrixTracker,
)
from sen1floods11 import Sen1Floods11Dataset, download_sen1floods11


def evaluate(model, loader, criterion, device):
    """
    Evaluate model over a DataLoader using exact confusion matrix tracking.

    Returns
    -------
    avg_loss : float
    metrics : dict (iou, f1, precision, recall, accuracy, tp, fp, fn, tn)
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


def main():
    parser = argparse.ArgumentParser(description="Train Advanced Models on SAR Flood Datasets")
    parser.add_argument("--data-dir", default="data/sen1floods11", help="Path to Sen1Floods11 directory")
    parser.add_argument("--download", action="store_true", help="Download Sen1Floods11 data if not present")
    parser.add_argument("--arch", choices=["resnet34", "unet"], default="resnet34", help="Model architecture")
    parser.add_argument("--in-channels", type=int, default=3, choices=[2, 3], help="Input channels (2: [VV, VH], 3: [VV, VH, diff])")
    parser.add_argument("--include-perm", action="store_true", help="Include permanent water dataset chips in training")
    parser.add_argument("--loss", choices=["focal_tversky", "bce_dice"], default="focal_tversky", help="Loss function")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--output-dir", default="models", help="Directory to save checkpoints and plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Advanced Sentinel-1 Deep Learning Flood Training Pipeline")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # --- 1. Dataset Preparation / Download ---
    if args.download or not os.path.exists(os.path.join(args.data_dir, "splits")):
        print(f"\nVerifying / Downloading Sen1Floods11 dataset into '{args.data_dir}'...")
        download_sen1floods11(args.data_dir, include_perm=args.include_perm, workers=16)

    print("\nLoading dataset splits:")
    train_ds = Sen1Floods11Dataset(
        data_dir=args.data_dir, split="train", in_channels=args.in_channels,
        include_perm=args.include_perm, augment=True
    )
    val_ds = Sen1Floods11Dataset(
        data_dir=args.data_dir, split="val", in_channels=args.in_channels,
        include_perm=False, augment=False
    )
    test_ds = Sen1Floods11Dataset(
        data_dir=args.data_dir, split="test", in_channels=args.in_channels,
        include_perm=False, augment=False
    )

    if len(train_ds) == 0:
        print(f"[Error] No training chips found in {args.data_dir}. Please run with `--download` flag.", file=sys.stderr)
        sys.exit(1)

    print(f"  Train: {len(train_ds)} chips (Include Permanent Water: {args.include_perm})")
    print(f"  Val:   {len(val_ds)} chips")
    print(f"  Test:  {len(test_ds)} chips")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=(len(train_ds) > args.batch_size)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    # --- 2. Build Model & Loss ---
    model = get_model(arch=args.arch, in_channels=args.in_channels, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.arch.upper()} (in_channels={args.in_channels}, params={n_params:,})")

    if args.loss == "focal_tversky":
        # alpha=0.3, beta=0.7 puts high penalty on false negatives -> boosts recall
        criterion = MaskedFocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
        print("Loss: Masked Focal Tversky Loss (alpha=0.3, beta=0.7, gamma=1.33 - Optimized for High Recall)")
    else:
        criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        print("Loss: Masked BCE + Dice Loss (50/50)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- 3. Training Loop ---
    best_val_iou = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
    }

    print("\nStarting training:")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val IoU':>8} | {'Val F1':>8} | {'Precision':>9} | {'Recall':>8} | {'Time':>6}")
    print("-" * 88)

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
            logits = model(x)
            loss = criterion(logits, y, mask=mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss /= max(num_batches, 1)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
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
        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>9.4f} | {val_iou*100:>7.2f}% | {val_f1*100:>7.2f}% | {val_prec*100:>8.2f}% | {val_rec*100:>7.2f}% | {elapsed:>5.1f}s")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "arch": args.arch,
                "in_channels": args.in_channels,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
            }, best_path)

    total_time = time.time() - t_start
    print("-" * 88)
    print(f"Training completed in {total_time / 60:.1f} minutes. Best Val IoU: {best_val_iou * 100:.2f}%")

    # --- 4. Final Evaluation on Test Split ---
    print("\n" + "=" * 70)
    print("Evaluating Best Model on Official Sen1Floods11 Global Test Set:")
    print("=" * 70)
    best_checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pth"), map_location=device, weights_only=False)
    model = get_model(arch=best_checkpoint.get("arch", "unet"), in_channels=best_checkpoint.get("in_channels", 2)).to(device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Test Loss:       {test_loss:.4f}")
    print(f"  Test IoU:        {test_metrics['iou'] * 100:.2f}%")
    print(f"  Test F1 / Dice:  {test_metrics['f1'] * 100:.2f}%")
    print(f"  Test Precision:  {test_metrics['precision'] * 100:.2f}%")
    print(f"  Test Recall:     {test_metrics['recall'] * 100:.2f}%")
    print(f"  Test Accuracy:   {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Pixel Counts:    TP={test_metrics['tp']:,}, FP={test_metrics['fp']:,}, FN={test_metrics['fn']:,}, TN={test_metrics['tn']:,}")

    # --- 5. Save Final History & Plot Curves ---
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=3)
        ax1.plot(epochs_range, history["val_loss"], "r-s", label="Val Loss", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(epochs_range, [v * 100 for v in history["val_iou"]], "g-^", label="Val IoU (%)", markersize=3)
        ax2.plot(epochs_range, [v * 100 for v in history["val_f1"]], "m-d", label="Val F1 (%)", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score (%)")
        ax2.set_title("Validation IoU & F1 Scores")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nTraining curves saved to: {plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate plot: {e}")

    print(f"Checkpoints saved to: '{args.output_dir}/'")


if __name__ == "__main__":
    main()
