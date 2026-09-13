#!/usr/bin/env python3
"""
Full-Scene Sliding-Window Inference for Landslide Mapping

Processes large satellite scenes (potentially >10,000x10,000 pixels) using:
  - Sliding window: 512x512 patches with 256 stride (50% overlap)
  - 2D Hann cosine window blending to eliminate tile seam artifacts
  - 4-fold Test-Time Augmentation (TTA): original, hflip, vflip, rot90
  - Memory-safe chunked GPU processing

Direct port and adaptation of the flood pipeline's test_custom_s1.py approach.

Usage:
    python predict_fullscene.py --model models/best_model.pth \\
        --input data/raw/stacked_scene.tif \\
        --output output/wayanad_landslide_map.tif
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import rasterio
    from rasterio.enums import Resampling
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from models import get_model


# ---------------------------------------------------------------------------
# 1. 2D Hann window
# ---------------------------------------------------------------------------
def build_hann_window(size: int) -> np.ndarray:
    """
    Build a 2D Hann (cosine-tapered) blending window.

    Parameters
    ----------
    size : int
        Side length of the square window.

    Returns
    -------
    W : np.ndarray  shape (size, size), float32
        Window values in [1e-4, 1.0].  Edges taper to near-zero so that
        overlapping patches blend smoothly with no tile-seam artefacts.
    """
    w = np.hanning(size)                   # 1-D Hann vector, length = size
    W = np.outer(w, w)                     # 2-D via outer product
    W = np.maximum(W, 1e-4).astype(np.float32)
    return W


# ---------------------------------------------------------------------------
# 2. TTA predict for a single patch tensor
# ---------------------------------------------------------------------------
def tta_predict(model: nn.Module,
                patch_tensor: torch.Tensor,
                device: torch.device) -> np.ndarray:
    """
    4-fold Test-Time Augmentation inference on a single patch tensor.

    Augmentations applied and then reversed before averaging:
      1. Original
      2. Horizontal flip  (dims=-1)
      3. Vertical flip    (dims=-2)
      4. 90° rotation     (rot90)

    Parameters
    ----------
    model : nn.Module  (eval mode)
    patch_tensor : torch.Tensor  shape (1, C, H, W)
    device : torch.device

    Returns
    -------
    avg_prob : np.ndarray  shape (H, W), float32
    """
    model.eval()

    with torch.no_grad():
        # Augmentation 0: original
        p0 = patch_tensor.to(device)
        prob0 = torch.sigmoid(model(p0))                        # (1,1,H,W)

        # Augmentation 1: horizontal flip (left-right)
        p1 = torch.flip(patch_tensor, dims=[-1]).to(device)
        prob1 = torch.sigmoid(model(p1))
        prob1 = torch.flip(prob1, dims=[-1])                    # undo flip

        # Augmentation 2: vertical flip (up-down)
        p2 = torch.flip(patch_tensor, dims=[-2]).to(device)
        prob2 = torch.sigmoid(model(p2))
        prob2 = torch.flip(prob2, dims=[-2])                    # undo flip

        # Augmentation 3: 90° rotation (counter-clockwise)
        p3 = torch.rot90(patch_tensor, k=1, dims=[-2, -1]).to(device)
        prob3 = torch.sigmoid(model(p3))
        prob3 = torch.rot90(prob3, k=-1, dims=[-2, -1])         # undo rotation

        avg_prob = (prob0 + prob1 + prob2 + prob3) / 4.0       # (1,1,H,W)

    return avg_prob.squeeze().cpu().numpy().astype(np.float32)  # (H, W)


# ---------------------------------------------------------------------------
# 3. Sliding-window inference
# ---------------------------------------------------------------------------
def sliding_window_inference(
    model: nn.Module,
    image_array: np.ndarray,
    device: torch.device,
    patch_size: int = 512,
    stride: int = 256,
    batch_size: int = 4,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Full-scene sliding-window inference with 2D Hann blending.

    Parameters
    ----------
    model : nn.Module  (eval mode)
    image_array : np.ndarray
        Full-scene image, shape (H, W, C) or (C, H, W).
    device : torch.device
    patch_size : int
        Square patch size in pixels (default: 512).
    stride : int
        Sliding step; overlap = patch_size - stride (default: 256 = 50%).
    batch_size : int
        Number of patches processed per GPU forward pass (default: 4).
    use_tta : bool
        Enable 4-fold TTA (default: True).

    Returns
    -------
    prob_map : np.ndarray  shape (H, W), float32
        Blended pixel-wise landslide probability map.
    """
    # Normalise to (C, H, W)
    arr = image_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.transpose(arr, (2, 0, 1))   # (H, W, C) → (C, H, W)

    C, H, W = arr.shape

    prob_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)
    window = build_hann_window(patch_size)                      # (P, P)

    # --- Build list of patch coordinates ---
    y_steps: list[int] = list(range(0, H - patch_size + 1, stride))
    if not y_steps or (y_steps[-1] + patch_size) < H:
        y_steps.append(max(0, H - patch_size))

    x_steps: list[int] = list(range(0, W - patch_size + 1, stride))
    if not x_steps or (x_steps[-1] + patch_size) < W:
        x_steps.append(max(0, W - patch_size))

    coords = [(py, px) for py in y_steps for px in x_steps]
    total_patches = len(coords)

    print(
        f"\nScene size  : {W} × {H} pixels  ({C} channels)")
    print(
        f"Patch size  : {patch_size}  stride={stride}  "
        f"overlap={patch_size - stride}px")
    print(f"Total patches: {total_patches:,}  (batch_size={batch_size}, TTA={use_tta})")

    model.eval()

    # --- Iterate in batches ---
    pbar_iter = range(0, total_patches, batch_size)
    if _TQDM_AVAILABLE:
        pbar_iter = tqdm(pbar_iter, desc="Sliding-window inference",
                         unit="batch", total=(total_patches + batch_size - 1) // batch_size)

    for b_start in pbar_iter:
        batch_coords = coords[b_start: b_start + batch_size]

        # Build batch tensor
        patches = []
        for py, px in batch_coords:
            patch = arr[:, py: py + patch_size, px: px + patch_size]  # (C,P,P)
            patches.append(patch)
        batch_t = torch.from_numpy(np.stack(patches, axis=0))  # (B, C, P, P)

        if use_tta:
            # TTA: process per-patch (avoids transposing a whole batch)
            for i, (py, px) in enumerate(batch_coords):
                single_t = batch_t[i: i + 1]                  # (1, C, P, P)
                pred = tta_predict(model, single_t, device)    # (P, P)
                prob_sum[py: py + patch_size, px: px + patch_size] += pred * window
                weight_sum[py: py + patch_size, px: px + patch_size] += window
        else:
            batch_t = batch_t.to(device)
            with torch.no_grad():
                logits = model(batch_t)                        # (B, 1, P, P)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, P, P)
            for i, (py, px) in enumerate(batch_coords):
                prob_sum[py: py + patch_size, px: px + patch_size] += (
                    probs[i] * window
                )
                weight_sum[py: py + patch_size, px: px + patch_size] += window

    # Avoid division by zero in unvisited corners
    weight_sum = np.maximum(weight_sum, 1e-9)
    prob_map = (prob_sum / weight_sum).astype(np.float32)
    return prob_map


# ---------------------------------------------------------------------------
# 4. GeoTIFF I/O
# ---------------------------------------------------------------------------
def load_scene_as_array(geotiff_path: str) -> tuple[np.ndarray, dict]:
    """
    Read a multi-band GeoTIFF into a (C, H, W) float32 numpy array.

    Parameters
    ----------
    geotiff_path : str
        Path to a multi-band GeoTIFF.

    Returns
    -------
    image_array : np.ndarray  shape (C, H, W), float32
    profile : dict  rasterio dataset profile (CRS, transform, dtype …)
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for GeoTIFF I/O. "
            "Install with: pip install rasterio"
        )
    if not os.path.exists(geotiff_path):
        raise FileNotFoundError(f"GeoTIFF not found: '{geotiff_path}'")

    with rasterio.open(geotiff_path) as ds:
        image_array = ds.read().astype(np.float32)   # (C, H, W)
        profile = ds.profile.copy()

    print(
        f"Loaded scene: {os.path.basename(geotiff_path)}  "
        f"shape=({image_array.shape[0]}C × {image_array.shape[2]}W × {image_array.shape[1]}H)"
    )
    return image_array, profile


def save_probability_map(
    prob_map: np.ndarray,
    ref_profile: dict,
    output_path: str,
) -> None:
    """
    Save a float32 probability map as a single-band GeoTIFF, preserving the
    CRS, transform, and spatial extent from the reference profile.

    Parameters
    ----------
    prob_map : np.ndarray  shape (H, W), float32
    ref_profile : dict
        rasterio profile from the source scene (provides CRS, transform, etc.).
    output_path : str
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for GeoTIFF output.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_profile = ref_profile.copy()
    out_profile.update(
        {
            "count": 1,
            "dtype": "float32",
            "nodata": -9999.0,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    print(f"Probability map saved: {output_path}")


# ---------------------------------------------------------------------------
# 5. Post-processing helpers
# ---------------------------------------------------------------------------
def _postprocess_map(
    prob_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply threshold and minimal morphological clean-up to produce a binary
    landslide map.

    Parameters
    ----------
    prob_map : np.ndarray  shape (H, W)
    threshold : float

    Returns
    -------
    binary_map : np.ndarray  shape (H, W), uint8  — {0, 1}
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    # Optional: remove tiny isolated blobs (requires scipy)
    try:
        from scipy import ndimage
        # Label connected components and remove those smaller than 9 pixels
        labeled, n_comps = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(1, n_comps + 1))
        for comp_id, sz in enumerate(sizes, start=1):
            if sz < 9:
                binary[labeled == comp_id] = 0
    except ImportError:
        pass  # scipy not installed; skip morphological filtering

    return binary


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full-Scene Sliding-Window Landslide Inference"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained checkpoint .pth file"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to multi-band GeoTIFF scene"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the final landslide map GeoTIFF"
    )
    parser.add_argument(
        "--patch-size", type=int, default=512,
        help="Sliding-window patch size in pixels (default: 512)"
    )
    parser.add_argument(
        "--stride", type=int, default=256,
        help="Sliding-window stride in pixels (default: 256 = 50%% overlap)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Probability threshold for binary prediction (default: 0.5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of patches per GPU batch (default: 4)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cuda', 'cpu', or None for auto-detect"
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Disable 4-fold Test-Time Augmentation (faster but less accurate)"
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    if args.stride >= args.patch_size:
        print(
            f"[Warning] stride ({args.stride}) >= patch_size ({args.patch_size}). "
            "There will be gaps in coverage. Recommend stride < patch_size.",
            file=sys.stderr
        )

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_tta = not args.no_tta

    print("=" * 70)
    print("Landslide Full-Scene Sliding-Window Inference Pipeline")
    print("=" * 70)
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Patch size   : {args.patch_size}")
    print(f"Stride       : {args.stride}  (overlap: {args.patch_size - args.stride}px)")
    print(f"Threshold    : {args.threshold}")
    print(f"TTA          : {use_tta}")

    # --- 1. Load model ---
    if not os.path.exists(args.model):
        print(f"[Error] Checkpoint not found: '{args.model}'", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "resnet34_unet")
    in_channels = checkpoint.get("in_channels", 14)
    model = get_model(arch=arch, in_channels=in_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"\nLoaded {arch.upper()} checkpoint '{args.model}' "
        f"(in_channels={in_channels}, "
        f"Val IoU: {checkpoint.get('val_iou', 0.0) * 100:.2f}%)"
    )

    # --- 2. Load scene ---
    print("\n[1/4] Loading scene from GeoTIFF...")
    t0 = time.time()
    image_array, ref_profile = load_scene_as_array(args.input)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Scene statistics
    print(f"  Value range : [{image_array.min():.4f}, {image_array.max():.4f}]")
    print(f"  NaN pixels  : {int(np.isnan(image_array).sum()):,}")
    # Replace NaN with 0 to avoid propagation
    image_array = np.nan_to_num(image_array, nan=0.0)

    # --- 3. Sliding-window inference ---
    print("\n[2/4] Running sliding-window inference...")
    t0 = time.time()
    prob_map = sliding_window_inference(
        model=model,
        image_array=image_array,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        use_tta=use_tta,
    )
    inference_time = time.time() - t0
    print(f"\n  Inference completed in {inference_time:.1f}s "
          f"({inference_time / 60:.1f} min)")

    # --- 4. Save probability map ---
    print("\n[3/4] Saving probability map...")
    prob_out = os.path.splitext(args.output)[0] + "_prob.tif"
    save_probability_map(prob_map, ref_profile, prob_out)

    # --- 5. Post-process and save final binary map ---
    print("\n[4/4] Applying threshold and post-processing...")
    binary_map = _postprocess_map(prob_map, threshold=args.threshold)

    landslide_px = int(binary_map.sum())
    total_px = binary_map.size
    print(
        f"  Detected landslide pixels: {landslide_px:,} / {total_px:,} "
        f"({100.0 * landslide_px / total_px:.4f}%)"
    )

    # Pixel area estimate (assume 10m × 10m Sentinel resolution by default)
    H, W = prob_map.shape
    # Try to derive pixel resolution from profile transform
    try:
        transform = ref_profile.get("transform")
        px_m = abs(float(transform.a)) if transform else 10.0
        py_m = abs(float(transform.e)) if transform else 10.0
    except Exception:
        px_m = py_m = 10.0

    landslide_area_km2 = landslide_px * px_m * py_m / 1e6
    print(f"  Estimated landslide area: {landslide_area_km2:.3f} km²  "
          f"(at {px_m:.1f}m × {py_m:.1f}m resolution)")

    # Save binary map as uint8 GeoTIFF
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    binary_profile = ref_profile.copy()
    binary_profile.update(
        {"count": 1, "dtype": "uint8", "nodata": 255, "compress": "deflate"}
    )

    if _RASTERIO_AVAILABLE:
        with rasterio.open(args.output, "w", **binary_profile) as dst:
            dst.write(binary_map, 1)
        print(f"Binary landslide map saved: {args.output}")
    else:
        # Fallback: save as .npy if rasterio unavailable
        npy_out = os.path.splitext(args.output)[0] + "_binary.npy"
        np.save(npy_out, binary_map)
        print(f"[Warning] rasterio not available; binary map saved as: {npy_out}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Inference Statistics:")
    print("=" * 70)
    print(f"  Scene dimensions : {W} × {H} pixels")
    print(f"  Inference time   : {inference_time:.1f}s  ({inference_time / 60:.2f} min)")
    print(f"  Landslide pixels : {landslide_px:,}  ({100.0 * landslide_px / total_px:.4f}%)")
    print(f"  Landslide area   : {landslide_area_km2:.3f} km²")
    print(f"  Probability map  : {prob_out}")
    print(f"  Binary map       : {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
