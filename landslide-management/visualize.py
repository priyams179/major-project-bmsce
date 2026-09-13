#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Landslide Detection and Mapping

Provides:
  1. Patch-level comparison: Multi-panel view of RGB, SAR, Probability, and Ground Truth.
  2. Full-scene disaster maps: GeoTIFF overlays with color-coded classes.
  3. Bi-temporal change panels: Pre-event vs Post-event optical/SAR with change scars.
  4. Training history curves: Dual-panel Loss and IoU/F1 trajectory plots.
  5. Indian event summary charts: Cross-disaster impact visualization (Wayanad, Chamoli, etc.).

Usage:
    python visualize.py --type patch --image data/processed/l4s/test/image_00001.npy \\
                        --mask data/processed/l4s/test/mask_00001.npy --output figures/
    python visualize.py --type training --history models/training_history.json --output figures/
    python visualize.py --type fullscene --map output/landslide_response_map.tif --output figures/
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# Image Compositing Helpers
# ---------------------------------------------------------------------------
def make_rgb_composite(
    image_np: np.ndarray,
    rgb_indices: Tuple[int, int, int] = (2, 1, 0),
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Construct a displayable 3-channel RGB image with percentile contrast stretching.
    """
    # Accept (C, H, W) or (H, W, C)
    if image_np.ndim == 3 and image_np.shape[0] in (2, 3, 5, 6, 12, 14):
        image_np = np.transpose(image_np, (1, 2, 0))

    H, W, C = image_np.shape
    r_idx, g_idx, b_idx = rgb_indices

    r = image_np[:, :, min(r_idx, C - 1)].astype(np.float32)
    g = image_np[:, :, min(g_idx, C - 1)].astype(np.float32)
    b = image_np[:, :, min(b_idx, C - 1)].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)

    # Percentile stretch per band
    for c in range(3):
        band = rgb[:, :, c]
        valid = np.isfinite(band)
        if valid.any():
            vmin = np.percentile(band[valid], p_low)
            vmax = np.percentile(band[valid], p_high)
            if vmax > vmin:
                rgb[:, :, c] = np.clip((band - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                rgb[:, :, c] = np.zeros_like(band)

    return (rgb * 255).astype(np.uint8)


def make_false_color(
    image_np: np.ndarray,
    nir_idx: int = 3,
    red_idx: int = 2,
    green_idx: int = 1,
) -> np.ndarray:
    """
    Construct standard NIR-Red-Green false-color composite for vegetation analysis.
    Healthy canopy appears vivid red; stripped landslide scars appear cyan/gray.
    """
    return make_rgb_composite(image_np, rgb_indices=(nir_idx, red_idx, green_idx))


def overlay_landslide_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Overlay binary mask on RGB composite with transparency."""
    overlay = rgb.copy().astype(np.float32)
    idx = mask > 0
    overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-Panel Visualizations
# ---------------------------------------------------------------------------
def plot_patch_comparison(
    image_np: np.ndarray,
    prob_map: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    title: str = "Landslide Detection Analysis",
    output_path: Optional[str] = None,
) -> None:
    """
    Generate 4-panel comparison figure: RGB | False Color / SAR | Probability | Prediction vs GT.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    rgb = make_rgb_composite(image_np)
    false_col = make_false_color(image_np)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Panel 1: True Color RGB
    axes[0].imshow(rgb)
    axes[0].set_title("True Color (B4-B3-B2)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: False Color NIR
    axes[1].imshow(false_col)
    axes[1].set_title("False Color (NIR-R-G)", fontsize=11)
    axes[1].axis("off")

    # Panel 3: Neural Probability Map
    im_prob = axes[2].imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Landslide Probability", fontsize=11)
    axes[2].axis("off")
    fig.colorbar(im_prob, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Overlay / Comparison
    overlay = overlay_landslide_mask(rgb, pred_mask, alpha=0.55, color=(255, 30, 30))
    axes[3].imshow(overlay)
    if gt_mask is not None:
        axes[3].contour(gt_mask, levels=[0.5], colors=["yellow"], linewidths=1.5)
        axes[3].set_title("Prediction (Red) vs GT (Yellow)", fontsize=11)
    else:
        axes[3].set_title("Detected Landslide Overlay", fontsize=11)
    axes[3].axis("off")

    plt.suptitle(title, fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved patch comparison figure to: {output_path}")


def plot_temporal_comparison(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    change_mask: np.ndarray,
    event_name: str = "Disaster Event",
    output_path: Optional[str] = None,
) -> None:
    """
    3-panel bi-temporal comparison: Pre-Event | Post-Event | Delineated Change Scars.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    rgb_pre = make_rgb_composite(pre_img)
    rgb_post = make_rgb_composite(post_img)
    overlay = overlay_landslide_mask(rgb_post, change_mask, alpha=0.6, color=(255, 20, 20))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_pre)
    axes[0].set_title("Pre-Event Baseline", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(rgb_post)
    axes[1].set_title("Post-Event Scene", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Detected Landslides (Red)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(f"Bi-Temporal Landslide Mapping: {event_name}", fontsize=14, weight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved temporal comparison to: {output_path}")


def plot_training_curves(history_path: str, output_path: Optional[str] = None) -> None:
    """Plot training & validation loss and metric trajectories from JSON history."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-s", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Focal-Tversky Loss Trajectory", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # IoU / F1 subplot
    ax2.plot(epochs, [v * 100 for v in history["val_iou"]], "g-^", markersize=3, label="Val IoU (%)")
    ax2.plot(epochs, [v * 100 for v in history["val_f1"]], "m-d", markersize=3, label="Val F1 (%)")
    if "val_recall" in history:
        ax2.plot(epochs, [v * 100 for v in history["val_recall"]], "c-x", markersize=3, label="Val Recall (%)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score (%)", fontsize=11)
    ax2.set_title("Validation Metrics (IoU / F1 / Recall)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.suptitle("Deep Learning Landslide Segmentation Training Curves", fontsize=13, weight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Saved training curves to: {output_path}")


def plot_fullscene_map(landslide_tif: str, output_path: Optional[str] = None) -> None:
    """Plot georeferenced multi-class response map with legend."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    try:
        import rasterio
        with rasterio.open(landslide_tif) as ds:
            data = ds.read(1)
    except Exception as e:
        print(f"Could not read {landslide_tif}: {e}")
        return

    # Map classes: 0=Stable, 1=Landslide, 2=Bare Rock, 255=Excluded
    cmap = ListedColormap(["#228B22", "#DC143C", "#FF8C00", "#D3D3D3"])
    display_arr = np.zeros_like(data, dtype=np.uint8)
    display_arr[data == 0] = 0
    display_arr[data == 1] = 1
    display_arr[data == 2] = 2
    display_arr[data == 255] = 3

    plt.figure(figsize=(10, 8))
    plt.imshow(display_arr, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"Operational Landslide Mapping: {os.path.basename(landslide_tif)}", fontsize=13, pad=12)
    plt.axis("off")

    # Legend patches
    patches = [
        mpatches.Patch(color="#228B22", label="Stable Terrain"),
        mpatches.Patch(color="#DC143C", label="Newly Triggered Landslide"),
        mpatches.Patch(color="#FF8C00", label="Pre-Existing Bare Rock/Soil"),
        mpatches.Patch(color="#D3D3D3", label="Excluded (Slope <5°/>70° or Nodata)"),
    ]
    plt.legend(handles=patches, loc="lower right", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved full-scene visualization to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize Landslide Detection Predictions and Metrics")
    parser.add_argument("--type", required=True, choices=["patch", "temporal", "training", "fullscene"], help="Visualization type")
    parser.add_argument("--image", default=None, help="Input patch .npy file")
    parser.add_argument("--mask", default=None, help="Ground truth mask .npy file")
    parser.add_argument("--pred", default=None, help="Model prediction .npy or .tif file")
    parser.add_argument("--history", default="models/training_history.json", help="Training history JSON file")
    parser.add_argument("--map", default="output/landslide_response_map.tif", help="Full-scene response map GeoTIFF")
    parser.add_argument("--output", default="figures/output_plot.png", help="Output figure path")
    args = parser.parse_args()

    if args.type == "training":
        plot_training_curves(args.history, args.output)
    elif args.type == "fullscene":
        plot_fullscene_map(args.map, args.output)
    elif args.type == "patch":
        if not args.image:
            print("[Error] --image is required for patch visualization", file=sys.stderr)
            sys.exit(1)
        img = np.load(args.image)
        gt = np.load(args.mask) if args.mask else None

        # If prediction not provided, simulate or threshold probability
        if args.pred:
            prob = np.load(args.pred)
        else:
            prob = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32) if img.ndim == 3 else np.zeros_like(img[:, :, 0])
        pred_mask = (prob >= 0.5).astype(np.uint8)

        plot_patch_comparison(img, prob, pred_mask, gt_mask=gt, output_path=args.output)


if __name__ == "__main__":
    main()

