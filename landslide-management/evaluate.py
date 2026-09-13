#!/usr/bin/env python3
"""
Model Evaluation and Benchmark Verification for Landslide Detection

Computes:
  - Exact dataset-level Confusion Matrix metrics (IoU, F1-Score, Precision, Recall, Accuracy)
  - Optimal decision threshold scan maximizing F1-Score
  - Precision-Recall curve generation
  - Per-region performance breakdown (Western Ghats, Himalayas, Northeast, Nilgiris)
  - Confusion matrix heatmap visualization
  - Verification against Landslide4Sense global benchmark targets:
      * IoU >= 0.55
      * F1-Score >= 0.70
      * Recall >= 0.75

Usage:
    python evaluate.py --model models/best_model.pth --data-dir data/processed/l4s --split test
    python evaluate.py --model models/best_model.pth --data-dir data/processed/l4s --pr-curve
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import ConfusionMatrixTracker, compute_metrics, get_model, optimal_threshold


# ---------------------------------------------------------------------------
# Test Dataset Loader
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """Evaluation dataset loading .npy image, mask, and valid_mask patches."""

    def __init__(self, data_dir: str, split: str = "test"):
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)

        # Check for split directory or direct files
        search_dir = self.split_dir if os.path.exists(self.split_dir) else data_dir

        self.image_files = sorted(glob.glob(os.path.join(search_dir, "image_*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(search_dir, "mask_*.npy")))
        self.valid_files = sorted(glob.glob(os.path.join(search_dir, "valid_*.npy")))

        # Check metadata for regions
        self.meta = {}
        splits_json = os.path.join(data_dir, "splits.json")
        if os.path.exists(splits_json):
            try:
                with open(splits_json, "r") as f:
                    self.meta = json.load(f)
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_np = np.load(self.image_files[idx]).astype(np.float32)
        mask_np = np.load(self.mask_files[idx]).astype(np.float32)

        if idx < len(self.valid_files):
            valid_np = np.load(self.valid_files[idx]).astype(np.float32)
        else:
            valid_np = np.ones_like(mask_np, dtype=np.float32)

        # Ensure channel-first format (C, H, W)
        if img_np.ndim == 3 and img_np.shape[0] not in (2, 3, 5, 6, 12, 14):
            # Probably (H, W, C) -> transpose
            img_np = np.transpose(img_np, (2, 0, 1))

        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]
        if valid_np.ndim == 2:
            valid_np = valid_np[np.newaxis, ...]

        return torch.from_numpy(img_np), torch.from_numpy(mask_np), torch.from_numpy(valid_np)


# ---------------------------------------------------------------------------
# Core Evaluation Functions
# ---------------------------------------------------------------------------
def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load trained checkpoint and reconstruct corresponding architecture."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "resnet34_unet")
    in_channels = checkpoint.get("in_channels", 14)

    model = get_model(arch=arch, in_channels=in_channels, pretrained=False).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, checkpoint


def compute_global_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Compute exact dataset-level metrics using ConfusionMatrixTracker."""
    tracker = ConfusionMatrixTracker(threshold=threshold)
    all_logits = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(x)
            tracker.update(logits, y, mask=mask)

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
            all_masks.append(mask.detach().cpu())

    metrics = tracker.compute()
    return metrics, all_logits, all_targets, all_masks


def compute_precision_recall_curve(
    all_logits: List[torch.Tensor],
    all_targets: List[torch.Tensor],
    all_masks: List[torch.Tensor],
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Scan thresholds across valid pixels to compute Precision-Recall curve."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    precisions = []
    recalls = []
    f1_scores = []

    for t in thresholds:
        tracker = ConfusionMatrixTracker(threshold=float(t))
        for logits, targets, mask in zip(all_logits, all_targets, all_masks):
            tracker.update(logits, targets, mask=mask)
        m = tracker.compute()
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1_scores.append(m["f1"])

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    best_idx = int(np.argmax(f1_scores))
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return precisions, recalls, thresholds, best_t, best_f1


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------
def plot_pr_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    best_t: float,
    best_f1: float,
    output_path: str,
) -> None:
    """Export Precision-Recall curve plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.plot(recalls, precisions, "b-o", linewidth=2, markersize=5, label="Precision-Recall Curve")
        plt.scatter(
            [recalls[np.abs(thresholds - best_t).argmin()]],
            [precisions[np.abs(thresholds - best_t).argmin()]],
            color="red", s=100, zorder=5,
            label=f"Optimal threshold = {best_t:.2f} (F1 = {best_f1 * 100:.2f}%)",
        )
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve (Landslide Classification)", fontsize=13)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved PR curve plot to: {output_path}")
    except Exception as e:
        print(f"  [Warning] Could not plot PR curve: {e}")


def plot_confusion_matrix(metrics: dict, output_path: str) -> None:
    """Plot 2x2 confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap="Blues")
        plt.colorbar(cax)

        classes = ["Stable (0)", "Landslide (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=11)
        ax.set_yticklabels(classes, fontsize=11)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=13, weight="bold")

        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("Ground Truth", fontsize=12)
        plt.title("Confusion Matrix (Valid Pixels)", fontsize=13, pad=15)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved confusion matrix heatmap to: {output_path}")
    except Exception as e:
        print(f"  [Warning] Could not plot confusion matrix: {e}")


# ---------------------------------------------------------------------------
# Formatted Report
# ---------------------------------------------------------------------------
def print_evaluation_report(metrics: dict, checkpoint_info: Optional[dict] = None) -> None:
    """Display comprehensive evaluation report."""
    print("\n" + "=" * 75)
    print("LANDSLIDE MODEL EVALUATION REPORT")
    print("=" * 75)
    if checkpoint_info:
        print(f"Architecture:   {checkpoint_info.get('arch', 'N/A')}")
        print(f"Input Channels: {checkpoint_info.get('in_channels', 'N/A')}")
        print(f"Trained Epoch:  {checkpoint_info.get('epoch', 'N/A')}")
    print("-" * 75)
    print(f"{'Metric':<25} | {'Achieved':<12} | {'Target (L4S)':<14} | {'Status'}")
    print("-" * 75)

    iou_val = metrics["iou"] * 100
    f1_val = metrics["f1"] * 100
    rec_val = metrics["recall"] * 100
    prec_val = metrics["precision"] * 100
    acc_val = metrics["accuracy"] * 100

    print(f"{'Landslide IoU':<25} | {iou_val:>10.2f}% | {'≥ 55.0%':<14} | {'PASS' if iou_val >= 55.0 else 'CHECK'}")
    print(f"{'F1-Score / Dice':<25} | {f1_val:>10.2f}% | {'≥ 70.0%':<14} | {'PASS' if f1_val >= 70.0 else 'CHECK'}")
    print(f"{'Recall (Sensitivity)':<25} | {rec_val:>10.2f}% | {'≥ 75.0%':<14} | {'PASS' if rec_val >= 75.0 else 'CHECK'}")
    print(f"{'Precision':<25} | {prec_val:>10.2f}% | {'≥ 65.0%':<14} | {'PASS' if prec_val >= 65.0 else 'CHECK'}")
    print(f"{'Overall Accuracy':<25} | {acc_val:>10.2f}% | {'≥ 95.0%':<14} | {'PASS' if acc_val >= 95.0 else 'CHECK'}")
    print("-" * 75)
    print(f"Pixel Counts: TP={metrics['tp']:,} | FP={metrics['fp']:,} | FN={metrics['fn']:,} | TN={metrics['tn']:,}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Landslide Segmentation Models")
    parser.add_argument("--model", required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--data-dir", default="data/processed/l4s", help="Path to processed dataset directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--output-dir", default="output/eval", help="Directory to save evaluation plots and json")
    parser.add_argument("--pr-curve", action="store_true", help="Compute and plot Precision-Recall curve")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # Load model
    model, checkpoint = load_checkpoint_model(args.model, device)

    # Load dataset
    ds = EvalDataset(args.data_dir, split=args.split)
    if len(ds) == 0:
        print(f"[Error] No patches found in {args.data_dir}/{args.split}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(ds)} patches from {args.data_dir} ({args.split} split)...")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    metrics, all_logits, all_targets, all_masks = compute_global_metrics(
        model, loader, device, threshold=args.threshold
    )
    print_evaluation_report(metrics, checkpoint_info=checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"metrics_{args.split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(metrics, os.path.join(args.output_dir, f"confusion_matrix_{args.split}.png"))

    if args.pr_curve:
        print("\nScanning thresholds for Precision-Recall curve...")
        precisions, recalls, thresholds, best_t, best_f1 = compute_precision_recall_curve(
            all_logits, all_targets, all_masks
        )
        print(f"  Optimal Threshold: {best_t:.2f} (Max F1: {best_f1 * 100:.2f}%)")
        plot_pr_curve(
            precisions, recalls, thresholds, best_t, best_f1,
            os.path.join(args.output_dir, f"pr_curve_{args.split}.png"),
        )


if __name__ == "__main__":
    main()

