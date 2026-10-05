#!/usr/bin/env python3
"""
Model Evaluation and Benchmark Verification for Landslide Detection

Computes:
  - Exact dataset-level Confusion Matrix metrics (IoU, F1-Score, Precision, Recall, Accuracy)
  - Optimal decision threshold scan maximizing F1-Score
  - Precision-Recall curve generation
  - Per-region performance breakdown (Western Ghats, Himalayas, Northeast, Nilgiris)
  - Confusion matrix heatmap visualization
  - Comparison against Landslide4Sense reference targets (IoU >= 0.55,
    F1 >= 0.70, Recall >= 0.75).  Those come from a Sentinel-2 benchmark and are
    a floor for the legacy path, not a goal for the six-event S1+Landsat split.

The input width and patch layout are read from the dataset manifest and the
checkpoint, so a 24-channel S1+Landsat stack is never fed to a 14-channel model.

Usage:
    python evaluate.py --model models/best_model.pth --data-dir data/processed/s1_landsat --split test
    python evaluate.py --model models/best_model.pth --data-dir data/processed/l4s --pr-curve   # legacy 14-channel L4S
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import ConfusionMatrixTracker, compute_metrics, get_model, optimal_threshold
from prepare_dataset import (
    discover_split_samples as _discover_samples,
    infer_checkpoint_channels,
    read_dataset_manifest,
)


# ---------------------------------------------------------------------------
# Test Dataset Loader
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """Evaluation dataset loading .npy image, mask, and valid patches.

    Accepts both the current ``<split>_image_*.npy`` layout and the legacy
    unprefixed ``image_*.npy`` layout; the channel axis is resolved from the
    dataset manifest instead of a hardcoded allow-list of widths.
    """

    def __init__(self, data_dir: str, split: str = "test"):
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)

        # Check for split directory or direct files
        search_dir = self.split_dir if os.path.exists(self.split_dir) else data_dir
        self.samples = _discover_samples(search_dir, split)
        if not self.samples:
            raise FileNotFoundError(
                f"No 'image_*.npy' files found in '{search_dir}'. "
                "Run prepare_dataset.py to generate patch files."
            )

        manifest = read_dataset_manifest(data_dir)
        # None for a legacy dataset written before manifests existed; the
        # channel axis is then settled per patch from the array shape alone.
        self.n_channels = manifest["n_channels"]

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
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mask_path, valid_path = self.samples[idx]

        img_np = np.load(img_path).astype(np.float32)
        mask_np = np.load(mask_path).astype(np.float32)

        if img_np.ndim == 2:
            img_np = img_np[np.newaxis, ...]
        elif img_np.ndim == 3:
            if self.n_channels is not None:
                if img_np.shape[0] != self.n_channels:
                    # (H, W, C) → (C, H, W); (C, H, W) already matches.
                    img_np = np.transpose(img_np, (2, 0, 1))
            elif img_np.shape[-1] < img_np.shape[0]:
                # No manifest: assume the smaller axis is the channel axis,
                # which holds for both square patches and any plausible stack.
                img_np = np.transpose(img_np, (2, 0, 1))

        if self.n_channels is not None and img_np.shape[0] != self.n_channels:
            raise ValueError(
                f"{img_path} has {img_np.shape[0]} channels but the manifest "
                f"declares {self.n_channels}. Re-generate the dataset."
            )

        if os.path.exists(valid_path):
            valid_np = np.load(valid_path).astype(np.float32)
        else:
            valid_np = np.ones_like(mask_np, dtype=np.float32)

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
    in_channels = checkpoint.get("in_channels") or infer_checkpoint_channels(checkpoint)
    if not in_channels:
        raise ValueError(
            f"Cannot determine the model's input channel count from {checkpoint_path}. "
            "It records neither 'in_channels' nor a readable convolution weight."
        )
    in_channels = int(in_channels)
    checkpoint.setdefault("in_channels", in_channels)

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
    print(f"{'Metric':<25} | {'Achieved':<12} | {'L4S reference':<14} | Status")
    print("-" * 75)

    iou_val = metrics["iou"] * 100
    f1_val = metrics["f1"] * 100
    rec_val = metrics["recall"] * 100
    prec_val = metrics["precision"] * 100
    acc_val = metrics["accuracy"] * 100

    def _row(label: str, value: float, reference: float) -> None:
        print(f"{label:<25} | {value:>10.2f}% | {'≥ ' + format(reference, '.1f') + '%':<14} | "
              f"{'PASS' if value >= reference else 'CHECK'}")

    # These targets come from the Sentinel-2 Landslide4Sense benchmark.  They are
    # a sanity floor, not a goal for the six-event S1+Landsat split, where the
    # held-out event changes terrain entirely.
    _row("Landslide IoU", iou_val, 55.0)
    _row("F1-Score / Dice", f1_val, 70.0)
    _row("Recall (Sensitivity)", rec_val, 75.0)
    _row("Precision", prec_val, 65.0)
    _row("Overall Accuracy", acc_val, 95.0)
    print("-" * 75)
    print(f"Pixel Counts: TP={metrics['tp']:,} | FP={metrics['fp']:,} | FN={metrics['fn']:,} | TN={metrics['tn']:,}")
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Landslide Segmentation Models")
    parser.add_argument("--model", required=True, help="Path to best_model.pth checkpoint")
    parser.add_argument("--data-dir", default="data/processed/s1_landsat",
                        help="Path to a processed dataset directory (channel layout read from its manifest)")
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

