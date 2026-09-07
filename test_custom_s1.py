#!/usr/bin/env python3
"""
End-to-End Testing Pipeline for Raw Sentinel-1 (.SAFE / .zip) from NASA Earthdata.

Does NOT require ESA SNAP. Automatically performs:
  1. Extraction of raw measurement TIFFs & calibration XMLs for VV + VH.
  2. Radiometric calibration (DN -> Sigma0 dB) and thermal noise removal.
  3. Sliding-window U-Net flood inference with Hann window blending.
  4. DEM slope masking (> 5.0 deg) to eliminate mountain radar shadows.
  5. Permanent water separation (if pre-flood scene provided).
  6. Exports multi-class Disaster Map GeoTIFF and high-res visualization figure.

Usage:
    uv run python test_custom_s1.py --post-flood data/S1A_...20240706...SAFE
    uv run python test_custom_s1.py --post-flood data/post.SAFE --pre-flood data/pre.SAFE
"""

import argparse
import glob
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import RegularGridInterpolator
from scipy.signal.windows import hann
import torch

from cnn_model import UNet
from postprocess import compute_terrain_slope, fetch_dem_for_raster, generate_disaster_response_map
from sen1floods11 import normalize_s1_band


# ---------------------------------------------------------------------------
# 1. SAFE Helpers & Radiometric Calibration
# ---------------------------------------------------------------------------
def ensure_extracted_safe(path: str) -> str:
    """If path is a .zip file, extract it to a .SAFE folder and return its directory."""
    if path.endswith(".zip"):
        extract_dir = os.path.dirname(path) or "."
        print(f"Extracting zip archive '{path}'...")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        safe_dirs = glob.glob(os.path.join(extract_dir, "*.SAFE"))
        if not safe_dirs:
            raise FileNotFoundError(f"No .SAFE folder found inside {path}")
        return safe_dirs[0]
    return path


def find_polarization_files(safe_dir: str, pol: str):
    """Find measurement TIFF, calibration XML, and noise XML for a polarization (vv or vh)."""
    pol_lower = pol.lower()
    tiffs = glob.glob(os.path.join(safe_dir, "measurement", f"*{pol_lower}*.tiff")) + \
            glob.glob(os.path.join(safe_dir, "measurement", f"*{pol_lower}*.tif"))
    if not tiffs:
        raise FileNotFoundError(f"No {pol.upper()} measurement TIFF found in {safe_dir}/measurement/")

    cal_xmls = glob.glob(os.path.join(safe_dir, "annotation", "calibration", f"calibration-*{pol_lower}*.xml"))
    noise_xmls = glob.glob(os.path.join(safe_dir, "annotation", "calibration", f"noise-*{pol_lower}*.xml"))

    cal_xml = cal_xmls[0] if cal_xmls else None
    noise_xml = noise_xmls[0] if noise_xmls else None
    return tiffs[0], cal_xml, noise_xml


def parse_xml_lut(xml_path: str, list_tag: str, possible_tags: list):
    """Parse calibration or noise LUT vectors from Sentinel-1 XML supporting multiple schema versions."""
    if not xml_path or not os.path.exists(xml_path):
        return None, None, None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        vlist = root.find(f".//{list_tag}")
        if vlist is None:
            return None, None, None

        lines = []
        pixels = None
        values = []
        for vec in vlist:
            line_elem = vec.find("line")
            if line_elem is None or not line_elem.text:
                continue
            line_val = float(line_elem.text)

            pix_elem = vec.find("pixel")
            if pix_elem is None or not pix_elem.text:
                continue
            pix_vals = np.array([float(x) for x in pix_elem.text.split()], dtype=np.float64)
            if pixels is None:
                pixels = pix_vals

            val_elem = None
            for tag in possible_tags:
                elem = vec.find(tag)
                if elem is not None and elem.text:
                    val_elem = elem
                    break
            if val_elem is None or not val_elem.text:
                continue

            val_vals = np.array([float(x) for x in val_elem.text.split()], dtype=np.float64)
            if len(val_vals) != len(pixels):
                continue
            lines.append(line_val)
            values.append(val_vals)

        if not lines or pixels is None or not values:
            return None, None, None

        return np.array(lines, dtype=np.float64), pixels, np.array(values, dtype=np.float64)
    except Exception as e:
        print(f"    [Warning] XML parse warning for {list_tag}: {e}")
        return None, None, None


def interpolate_lut(lines, pixels, values, n_lines, n_pixels):
    """Interpolate 2D LUT to full raster dimensions."""
    interp = RegularGridInterpolator(
        (lines, pixels), values, method="linear", bounds_error=False, fill_value=None
    )
    LL, PP = np.meshgrid(np.arange(n_lines, dtype=np.float64), np.arange(n_pixels, dtype=np.float64), indexing="ij")
    points = np.column_stack([LL.ravel(), PP.ravel()])
    return interp(points).reshape(n_lines, n_pixels).astype(np.float32)


def calibrate_safe_band(safe_dir: str, pol: str) -> np.ndarray:
    """Read and radiometrically calibrate a Sentinel-1 polarization band to dB."""
    tiff_path, cal_xml, noise_xml = find_polarization_files(safe_dir, pol)
    print(f"  Calibrating {pol.upper()} from {os.path.basename(tiff_path)}...")

    with rasterio.open(tiff_path) as ds:
        dn = ds.read(1).astype(np.float32)

    n_lines, n_pixels = dn.shape

    if cal_xml:
        c_lines, c_pixels, c_vals = parse_xml_lut(cal_xml, "calibrationVectorList", ["sigmaNought", "betaNought", "gamma"])
        if c_lines is not None:
            cal_lut = interpolate_lut(c_lines, c_pixels, c_vals, n_lines, n_pixels)
        else:
            cal_lut = np.ones((n_lines, n_pixels), dtype=np.float32)
    else:
        cal_lut = np.ones((n_lines, n_pixels), dtype=np.float32)

    if noise_xml:
        n_lines_lut, n_pixels_lut, n_vals = parse_xml_lut(noise_xml, "noiseRangeVectorList", ["noiseRangeLut", "noiseRangeLinearValue", "noiseLut"])
        if n_lines_lut is not None:
            noise_lut = interpolate_lut(n_lines_lut, n_pixels_lut, n_vals, n_lines, n_pixels)
        else:
            noise_lut = np.zeros((n_lines, n_pixels), dtype=np.float32)
    else:
        noise_lut = np.zeros((n_lines, n_pixels), dtype=np.float32)

    power = np.maximum(dn**2 - noise_lut, 0.0) / (cal_lut**2 + 1e-7)
    db = 10.0 * np.log10(np.maximum(power, 1e-6))
    return np.clip(db, -35.0, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Sliding Window Inference with 2D Hann Blending & TTA
# ---------------------------------------------------------------------------
def get_hann_window_2d(patch_size: int = 512) -> np.ndarray:
    """Create a 2D Hann (cosine) tapered window for smooth boundary blending."""
    w1 = hann(patch_size, sym=True)
    w2 = hann(patch_size, sym=True)
    w2d = np.outer(w1, w2)
    return np.maximum(w2d, 1e-4).astype(np.float32)


def predict_patch_tta(model, patch_tensor: torch.Tensor, use_tta: bool = False) -> torch.Tensor:
    """Run model inference with optional 4-fold Test-Time Augmentation."""
    logits = model(patch_tensor)
    prob = torch.sigmoid(logits)
    if not use_tta:
        return prob

    # 1. Horizontal Flip
    p_h = torch.flip(patch_tensor, dims=[-1])
    prob_h = torch.flip(torch.sigmoid(model(p_h)), dims=[-1])

    # 2. Vertical Flip
    p_v = torch.flip(patch_tensor, dims=[-2])
    prob_v = torch.flip(torch.sigmoid(model(p_v)), dims=[-2])

    # 3. 90-degree Rotation
    p_r = torch.rot90(patch_tensor, 1, dims=[-2, -1])
    prob_r = torch.rot90(torch.sigmoid(model(p_r)), -1, dims=[-2, -1])

    return (prob + prob_h + prob_v + prob_r) / 4.0


def predict_full_raster(model, vv_db: np.ndarray, vh_db: np.ndarray, device,
                        in_channels: int = 2, patch_size: int = 512,
                        stride: int = 256, use_tta: bool = False) -> np.ndarray:
    """Run sliding-window inference on large SAR raster with 2D Hann blending and TTA."""
    H, W = vv_db.shape
    vv_norm = normalize_s1_band(vv_db)
    vh_norm = normalize_s1_band(vh_db)

    if in_channels == 3:
        diff_norm = np.clip((vh_norm - vv_norm + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        X = np.stack([vv_norm, vh_norm, diff_norm], axis=0)
    else:
        X = np.stack([vv_norm, vh_norm], axis=0)

    prob_accum = np.zeros((H, W), dtype=np.float32)
    weight_accum = np.zeros((H, W), dtype=np.float32)
    window = get_hann_window_2d(patch_size)

    y_steps = list(range(0, H - patch_size + 1, stride))
    if y_steps[-1] + patch_size < H:
        y_steps.append(H - patch_size)

    x_steps = list(range(0, W - patch_size + 1, stride))
    if x_steps[-1] + patch_size < W:
        x_steps.append(W - patch_size)

    total_patches = len(y_steps) * len(x_steps)
    done = 0

    model.eval()
    with torch.no_grad():
        for py in y_steps:
            for px in x_steps:
                patch = X[:, py:py+patch_size, px:px+patch_size]
                t_in = torch.from_numpy(patch).unsqueeze(0).to(device)
                pred_patch = predict_patch_tta(model, t_in, use_tta=use_tta).squeeze().cpu().numpy()

                prob_accum[py:py+patch_size, px:px+patch_size] += pred_patch * window
                weight_accum[py:py+patch_size, px:px+patch_size] += window
                done += 1
                if done % 10 == 0 or done == total_patches:
                    print(f"  Inference Progress: {done}/{total_patches} patches ({(done/total_patches)*100:.1f}%)", end="\r")

    print(f"\n  Inference completed across {done} patches (TTA={use_tta}).")
    weight_accum = np.maximum(weight_accum, 1e-6)
    return (prob_accum / weight_accum).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Main Testing Routine
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test raw Sentinel-1 (.SAFE) from NASA Earthdata with U-Net")
    parser.add_argument("--post-flood", required=True, help="Path to post-flood .SAFE folder or .zip")
    parser.add_argument("--pre-flood", default=None, help="Path to pre-flood .SAFE folder or .zip (for permanent water baseline)")
    parser.add_argument("--model", default="models/best_model.pth", help="Path to trained U-Net checkpoint")
    parser.add_argument("--tta", action="store_true", help="Enable 4-fold Test-Time Augmentation")
    parser.add_argument("--dem", default=None, help="Path to external DEM GeoTIFF (optional)")
    parser.add_argument("--max-slope", type=float, default=5.0, help="Slope threshold in degrees (default: 5.0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Water probability threshold (default: 0.5)")
    parser.add_argument("--output-dir", default="predictions_custom", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Sentinel-1 Automated Disaster Response Testing Pipeline")
    print("=" * 70)
    print(f"Device: {device}")

    # --- 1. Load Model with Auto-Architecture Detection ---
    from cnn_model import get_model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    in_channels = checkpoint.get("in_channels", 2)
    arch = checkpoint.get("arch", "unet")
    model = get_model(arch=arch, in_channels=in_channels, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {arch.upper()} checkpoint '{args.model}' (in_channels={in_channels}, Val IoU: {checkpoint.get('val_iou', 0.0)*100:.2f}%)")

    # --- 2. Process Post-Flood Scene ---
    post_safe = ensure_extracted_safe(args.post_flood)
    event_name = os.path.basename(post_safe).replace(".SAFE", "")
    print(f"\n[1/4] Calibrating Post-Flood Scene: {event_name}")
    t0 = time.time()
    post_vv = calibrate_safe_band(post_safe, "vv")
    post_vh = calibrate_safe_band(post_safe, "vh")
    H, W = post_vv.shape
    print(f"  Calibrated in {time.time() - t0:.1f}s. Image size: {W} x {H} pixels")

    # --- 3. Neural Network Inference ---
    print("\n[2/4] Running Neural Network Sliding-Window Inference...")
    t0 = time.time()
    prob_map = predict_full_raster(
        model, post_vv, post_vh, device, in_channels=in_channels,
        patch_size=512, stride=256, use_tta=args.tta
    )
    print(f"  Inference completed in {time.time() - t0:.1f}s.")

    # --- 4. Permanent Water Separation (if pre-flood provided) ---
    perm_water_mask = None
    if args.pre_flood:
        pre_safe = ensure_extracted_safe(args.pre_flood)
        print(f"\n[3/4] Calibrating & Evaluating Pre-Flood Baseline: {os.path.basename(pre_safe)}")
        pre_vv = calibrate_safe_band(pre_safe, "vv")
        pre_vh = calibrate_safe_band(pre_safe, "vh")
        pre_prob = predict_full_raster(
            model, pre_vv, pre_vh, device, in_channels=in_channels,
            patch_size=512, stride=256, use_tta=args.tta
        )
        perm_water_mask = pre_prob > args.threshold
        print(f"  Permanent baseline water: {int(perm_water_mask.sum()):,} pixels")

    # --- 5. DEM Slope Masking ---
    print("\n[4/4] Applying Topographic & Permanent Water Post-Processing...")
    vh_tiff, _, _ = find_polarization_files(post_safe, "vh")
    dem_fetched = fetch_dem_for_raster(vh_tiff)
    slope_deg = None
    if dem_fetched is not None and not np.isnan(dem_fetched).all():
        slope_deg = compute_terrain_slope(dem_fetched, pixel_size_x_m=10.0, pixel_size_y_m=10.0)
        print(f"  Auto-fetched elevation: Mean slope={slope_deg.mean():.1f}°, Steep (> {args.max_slope}°): {int((slope_deg > args.max_slope).sum()):,} px")

    # Assemble Disaster Response Map
    disaster_map, stats = generate_disaster_response_map(
        prob_map=prob_map,
        slope_deg=slope_deg,
        max_slope=args.max_slope,
        perm_water_mask=perm_water_mask,
        threshold=args.threshold,
    )

    print("\n" + "=" * 70)
    print("Operational Disaster Response Summary:")
    print("=" * 70)
    print(f"  Active Flood Inundation: {stats['flood_pixels']:,} pixels ({stats['flood_area_sqkm']:.2f} km²)")
    if stats["permanent_water_pixels"] > 0:
        print(f"  Permanent Water Bodies:  {stats['permanent_water_pixels']:,} pixels ({stats['permanent_water_area_sqkm']:.2f} km²)")
    if stats["slope_masked_pixels"] > 0:
        print(f"  Radar Shadows Excluded:  {stats['slope_masked_pixels']:,} pixels")

    # Save Output GeoTIFF
    out_tif = os.path.join(args.output_dir, f"{event_name}_disaster_map.tif")
    transform = from_origin(0, H * 10.0, 10.0, 10.0)
    with rasterio.open(
        out_tif, "w", driver="GTiff", height=H, width=W, count=1,
        dtype=rasterio.uint8, crs=None, transform=transform, compress="deflate", nodata=255
    ) as dst:
        dst.write(disaster_map, 1)
    print(f"\n  Saved Disaster Map GeoTIFF: {out_tif}")

    # Generate Visualization PNG (downsampled for fast viewing)
    vis_stride = max(H // 1000, 1)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Downsampled previews
    axes[0].imshow(post_vv[::vis_stride, ::vis_stride], cmap="gray", vmin=np.percentile(post_vv, 2), vmax=np.percentile(post_vv, 98))
    axes[0].set_title("Sentinel-1 SAR (VV)")
    axes[0].axis("off")

    axes[1].imshow(post_vh[::vis_stride, ::vis_stride], cmap="gray", vmin=np.percentile(post_vh, 2), vmax=np.percentile(post_vh, 98))
    axes[1].set_title("Sentinel-1 SAR (VH)")
    axes[1].axis("off")

    axes[2].imshow(prob_map[::vis_stride, ::vis_stride], cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("U-Net Water Probability")
    axes[2].axis("off")

    # Color Map: 0=Dark, 1=Red (Flood), 2=Blue (Permanent), 255=Grey (Slope)
    sub_map = disaster_map[::vis_stride, ::vis_stride]
    map_vis = np.zeros((*sub_map.shape, 3), dtype=np.float32)
    map_vis[sub_map == 0] = [0.12, 0.12, 0.12]
    map_vis[sub_map == 1] = [1.0, 0.15, 0.15]
    map_vis[sub_map == 2] = [0.1, 0.55, 1.0]
    map_vis[sub_map == 255] = [0.4, 0.35, 0.3]

    axes[3].imshow(map_vis)
    axes[3].set_title("Disaster Response Map\n(Red: Flood, Blue: Perm Water)")
    axes[3].axis("off")

    plt.tight_layout()
    out_png = os.path.join(args.output_dir, f"{event_name}_disaster_response.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  Saved Visualization Figure: {out_png}")
    print("=" * 70)


if __name__ == "__main__":
    main()

