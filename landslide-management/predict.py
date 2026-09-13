#!/usr/bin/env python3
"""
Single-Chip Inference for Landslide Detection

Runs trained segmentation model on individual image patches and generates
multi-panel visualization comparing input, prediction, and ground truth.

Usage:
    python predict.py --model models/best_model.pth \\
        --input data/processed/l4s/test/image_00001.npy \\
        --mask data/processed/l4s/test/mask_00001.npy \\
        --output output/predictions/
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from models import get_model


# ---------------------------------------------------------------------------
# 1. Model loader
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str,
               device: torch.device) -> tuple[nn.Module, dict]:
    """
    Load a trained segmentation checkpoint and reconstruct the model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pth checkpoint file.
    device : torch.device

    Returns
    -------
    model : nn.Module  (in eval mode, on device)
    checkpoint : dict  (full checkpoint dictionary)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_path}'"
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    arch = checkpoint.get("arch", "resnet34_unet")
    in_channels = checkpoint.get("in_channels", 14)

    model = get_model(arch=arch, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded {arch.upper()} checkpoint (in_channels={in_channels}, "
        f"epoch={checkpoint.get('epoch', '?')}, "
        f"Val IoU={checkpoint.get('val_iou', 0.0) * 100:.2f}%)"
    )
    return model, checkpoint


# ---------------------------------------------------------------------------
# 2. Patch inference
# ---------------------------------------------------------------------------
def predict_patch(model: nn.Module, image_np: np.ndarray,
                  device: torch.device,
                  threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a single image patch.

    Parameters
    ----------
    model : nn.Module  (already in eval mode)
    image_np : np.ndarray
        Input image with shape (H, W, C) or (C, H, W), dtype float32.
    device : torch.device
    threshold : float
        Probability cutoff for binary prediction (default: 0.5).

    Returns
    -------
    prob_map : np.ndarray  shape (H, W), float32  – sigmoid probability
    pred_binary : np.ndarray  shape (H, W), uint8  – {0, 1}
    """
    # Normalise to (C, H, W) if needed
    arr = image_np.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Detect (H, W, C) layout: last dim is channels when it's smallest
        if arr.shape[-1] < arr.shape[0] and arr.shape[-1] < arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
    # arr is now (C, H, W)

    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)                               # (1, 1, H, W)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy() # (H, W)

    prob_map = prob.astype(np.float32)
    pred_binary = (prob_map >= threshold).astype(np.uint8)
    return prob_map, pred_binary


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------
def visualize_prediction(
    image_np: np.ndarray,
    prob_map: np.ndarray,
    pred_binary: np.ndarray,
    gt_mask: np.ndarray | None = None,
    output_path: str | None = None,
) -> None:
    """
    Generate a multi-panel figure comparing input, probability map,
    binary prediction overlay, and optional ground truth.

    Parameters
    ----------
    image_np : np.ndarray
        Input image, shape (H, W, C) or (C, H, W).
    prob_map : np.ndarray
        Probability map, shape (H, W), values in [0, 1].
    pred_binary : np.ndarray
        Binary landslide prediction, shape (H, W), values {0, 1}.
    gt_mask : np.ndarray or None
        Ground-truth mask, shape (H, W), values {0, 1}.
    output_path : str or None
        If given, save the figure to this path; otherwise plt.show().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[Error] matplotlib is required for visualisation: {exc}",
              file=sys.stderr)
        return

    # ---------- Build RGB composite ----------
    # Ensure HWC layout for display
    img = image_np.astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[0] < img.shape[-1]:
        # (C, H, W) → (H, W, C)
        img = np.transpose(img, (1, 2, 0))

    C = img.shape[-1]
    if C >= 3:
        # For Sentinel-2: B4(red)=3, B3(green)=2, B2(blue)=1 (0-indexed: 2,1,0)
        # For 14-ch L4S: channels 0,1,2 are typically B2,B3,B4
        # We try the first three channels as a proxy
        rgb = img[:, :, :3].copy()
    else:
        # SAR fallback: replicate VV/VH into grayscale RGB
        rgb = np.stack([img[:, :, 0]] * 3, axis=-1)

    # Per-channel percentile stretch to [0, 1]
    for c in range(rgb.shape[-1]):
        lo = np.percentile(rgb[:, :, c], 2)
        hi = np.percentile(rgb[:, :, c], 98)
        if hi > lo:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo), 0.0, 1.0)
        else:
            rgb[:, :, c] = 0.0

    # ---------- Build binary prediction overlay (red on RGB) ----------
    overlay = rgb.copy()
    landslide_mask = pred_binary.astype(bool)
    overlay[landslide_mask, 0] = 1.0   # red channel full
    overlay[landslide_mask, 1] = 0.15  # suppress green
    overlay[landslide_mask, 2] = 0.15  # suppress blue

    # ---------- Layout ----------
    n_panels = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Panel 1: RGB composite
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Composite\n(Bands B4/B3/B2 or VV/VH)")
    axes[0].axis("off")

    # Panel 2: Probability map
    im = axes[1].imshow(prob_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("Landslide Probability Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Binary prediction overlay
    axes[2].imshow(overlay)
    landslide_pct = 100.0 * float(pred_binary.sum()) / pred_binary.size
    axes[2].set_title(
        f"Binary Prediction Overlay\n"
        f"(Red = Landslide, {landslide_pct:.2f}% of pixels)"
    )
    axes[2].axis("off")

    # Panel 4: Ground truth (optional)
    if gt_mask is not None:
        gt_rgb = rgb.copy()
        gt_land = gt_mask.astype(bool)
        gt_rgb[gt_land, 0] = 1.0
        gt_rgb[gt_land, 1] = 0.15
        gt_rgb[gt_land, 2] = 0.15
        gt_pct = 100.0 * float(gt_mask.sum()) / gt_mask.size
        axes[3].imshow(gt_rgb)
        axes[3].set_title(
            f"Ground Truth\n(Red = Landslide, {gt_pct:.2f}% of pixels)"
        )
        axes[3].axis("off")

    plt.suptitle("Landslide Detection — Single Patch Inference", fontsize=14)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualisation saved to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Single-Chip Landslide Inference and Visualisation"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained checkpoint .pth file"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input image .npy file (shape: H×W×C or C×H×W)"
    )
    parser.add_argument(
        "--mask", default=None,
        help="Path to ground truth mask .npy file (optional)"
    )
    parser.add_argument(
        "--output", default="output/predictions",
        help="Output directory for saving prediction files (default: output/predictions/)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binary prediction (default: 0.5)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'cpu', or None for auto-detect"
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output, exist_ok=True)

    # --- Load model ---
    model, checkpoint = load_model(args.model, device)

    # --- Load input ---
    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: '{args.input}'", file=sys.stderr)
        sys.exit(1)
    image_np = np.load(args.input).astype(np.float32)
    print(f"Loaded input: {args.input}  shape={image_np.shape}")

    # --- Load ground truth mask ---
    gt_mask = None
    if args.mask is not None:
        if not os.path.exists(args.mask):
            print(f"[Warning] Mask file not found: '{args.mask}'. "
                  "Proceeding without ground truth.")
        else:
            gt_mask = np.load(args.mask).astype(np.float32)
            if gt_mask.ndim == 3:
                gt_mask = gt_mask.squeeze(0)
            gt_mask = (gt_mask > 0.5).astype(np.float32)
            print(f"Loaded ground truth mask: {args.mask}  "
                  f"landslide_pixels={int(gt_mask.sum()):,}")

    # --- Inference ---
    print(f"\nRunning inference (threshold={args.threshold})...")
    prob_map, pred_binary = predict_patch(
        model, image_np, device, threshold=args.threshold
    )

    landslide_px = int(pred_binary.sum())
    total_px = pred_binary.size
    print(f"Predicted landslide pixels: {landslide_px:,} / {total_px:,} "
          f"({100.0 * landslide_px / total_px:.3f}%)")

    # --- Save outputs ---
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    np.save(os.path.join(args.output, f"{input_stem}_prob.npy"), prob_map)
    np.save(os.path.join(args.output, f"{input_stem}_pred.npy"), pred_binary)
    print(f"Probability map  saved: {os.path.join(args.output, input_stem + '_prob.npy')}")
    print(f"Binary pred map  saved: {os.path.join(args.output, input_stem + '_pred.npy')}")

    # --- Compute metrics if ground truth provided ---
    if gt_mask is not None:
        pred_bool = pred_binary.astype(bool)
        gt_bool = gt_mask.astype(bool)
        tp = int((pred_bool & gt_bool).sum())
        fp = int((pred_bool & ~gt_bool).sum())
        fn = int((~pred_bool & gt_bool).sum())
        tn = int((~pred_bool & ~gt_bool).sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print("\nPatch Metrics vs Ground Truth:")
        print(f"  IoU       : {iou * 100:.2f}%")
        print(f"  F1 / Dice : {f1 * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec * 100:.2f}%")
        print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # --- Visualise ---
    vis_path = os.path.join(args.output, f"{input_stem}_visualisation.png")
    visualize_prediction(
        image_np=image_np,
        prob_map=prob_map,
        pred_binary=pred_binary,
        gt_mask=gt_mask,
        output_path=vis_path,
    )


if __name__ == "__main__":
    main()
