#!/usr/bin/env python3
"""
Operational Disaster Response Flood Prediction & Mapping

Runs trained U-Net inference on Sentinel-1 SAR imagery with production-grade
post-processing:
  1. DEM Slope Masking: Eliminates mountain radar shadows (slope > 5.0 deg).
  2. Permanent Water Separation: Distinguishes permanent rivers/lakes from active flood inundation.
  3. Multi-Class Disaster Map Export & Visualization.

Usage:
    uv run python predict_cnn.py --model models/best_model.pth --chip-id USA_905409
    uv run python predict_cnn.py --model models/best_model.pth --input-s1 path/to/s1.tif --dem path/to/dem.tif
"""

import argparse
import glob
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from cnn_model import UNet
from postprocess import (
    compute_terrain_slope,
    fetch_dem_for_raster,
    generate_disaster_response_map,
    resample_raster_to_reference,
)
from sen1floods11 import load_chip_ids_from_csv, normalize_s1_band


def predict_sar_array(model, vv: np.ndarray, vh: np.ndarray, device, in_channels: int = 2) -> np.ndarray:
    """Run U-Net on VV and VH 2D arrays, returning 2D probability map."""
    vv_norm = normalize_s1_band(vv)
    vh_norm = normalize_s1_band(vh)

    if in_channels == 3:
        diff_norm = np.clip((vh_norm - vv_norm + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        x = np.stack([vv_norm, vh_norm, diff_norm], axis=0)
    else:
        x = np.stack([vv_norm, vh_norm], axis=0)

    tensor_in = torch.from_numpy(x).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_in)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    return prob


def save_multiclass_geotiff(path: str, data: np.ndarray, ref_geotiff_path: str, nodata: int = 255):
    """Save disaster response array as a standard GeoTIFF preserving CRS and transform."""
    with rasterio.open(ref_geotiff_path) as ref_ds:
        profile = ref_ds.profile.copy()

    profile.update(
        count=1,
        dtype=rasterio.uint8 if data.dtype == np.uint8 else rasterio.float32,
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description="Operational U-Net Flood Detection with DEM & Permanent Water Masking")
    parser.add_argument("--model", default="models/best_model.pth", help="Path to trained U-Net checkpoint")
    parser.add_argument("--data-dir", default="data/sen1floods11", help="Sen1Floods11 dataset directory")
    parser.add_argument("--chip-id", default=None, help="Chip ID to evaluate")
    parser.add_argument("--input-s1", default=None, help="Path to external 2-band S1 GeoTIFF (Band 1: VV, Band 2: VH)")
    parser.add_argument("--dem", default=None, help="Path to Digital Elevation Model (DEM) GeoTIFF")
    parser.add_argument("--max-slope", type=float, default=5.0, help="Max slope angle in degrees to filter mountain radar shadows (default: 5.0)")
    parser.add_argument("--pre-flood-s1", default=None, help="Path to pre-flood Sentinel-1 SAR raster to detect permanent water baseline")
    parser.add_argument("--permanent-water", default=None, help="Path to permanent water mask raster (e.g. JRC GSW)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for surface water")
    parser.add_argument("--output-dir", default="predictions", help="Output directory for predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("Operational Disaster Response Flood Mapping")
    print("=" * 65)
    print(f"Device: {device}")

    # --- 1. Load Model ---
    if not os.path.exists(args.model):
        print(f"[Error] Model checkpoint not found at '{args.model}'. Please train the model first.", file=sys.stderr)
        return

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    in_channels = checkpoint.get("in_channels", 2)
    model = UNet(in_channels=in_channels, base_features=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_iou = checkpoint.get("val_iou", 0.0)
    print(f"Loaded model from '{args.model}' (in_channels={in_channels}, Best Val IoU={val_iou*100:.2f}%)")

    # --- 2. Determine Input SAR Image ---
    s1_dir = os.path.join(args.data_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
    label_dir = os.path.join(args.data_dir, "v1.1", "data", "flood_events", "HandLabeled", "LabelHand")

    if args.input_s1:
        s1_path = args.input_s1
        chip_id = os.path.splitext(os.path.basename(s1_path))[0]
        label_path = None
    elif args.chip_id:
        chip_id = args.chip_id
        s1_path = os.path.join(s1_dir, f"{chip_id}_S1Hand.tif")
        label_path = os.path.join(label_dir, f"{chip_id}_LabelHand.tif")
    else:
        # Default: pick first chip from test split
        test_csv = os.path.join(args.data_dir, "splits", "flood_handlabeled", "flood_test_data.csv")
        if os.path.exists(test_csv):
            chips = load_chip_ids_from_csv(test_csv)
            chip_id = next((c for c in chips if os.path.exists(os.path.join(s1_dir, f"{c}_S1Hand.tif"))), None)
        else:
            chips = glob.glob(os.path.join(s1_dir, "*_S1Hand.tif"))
            chip_id = os.path.basename(chips[0]).replace("_S1Hand.tif", "") if chips else None

        if not chip_id:
            print("[Error] No SAR images found. Please download Sen1Floods11 or provide `--input-s1`.", file=sys.stderr)
            return
        s1_path = os.path.join(s1_dir, f"{chip_id}_S1Hand.tif")
        label_path = os.path.join(label_dir, f"{chip_id}_LabelHand.tif")

    print(f"\nProcessing Scene ID: {chip_id}")
    print(f"  Post-Flood SAR: {s1_path}")

    with rasterio.open(s1_path) as ds:
        vv = ds.read(1).astype(np.float32)
        vh = ds.read(2).astype(np.float32)
        bounds = ds.bounds
        crs = ds.crs
        res_x = abs(ds.transform[0])
        res_y = abs(ds.transform[4])

    H, W = vv.shape
    # Convert pixel resolution to metres if CRS is geographic
    if crs and crs.is_geographic:
        center_lat = (bounds.bottom + bounds.top) / 2.0
        pixel_size_y_m = res_y * 111320.0
        pixel_size_x_m = res_x * 111320.0 * math.cos(math.radians(center_lat))
    else:
        pixel_size_x_m = res_x if res_x > 0 else 10.0
        pixel_size_y_m = res_y if res_y > 0 else 10.0

    print(f"  Raster Size:    {W} x {H} pixels (Resolution: {pixel_size_x_m:.1f}m x {pixel_size_y_m:.1f}m)")

    # --- 3. Neural Network Inference ---
    t0 = time.time()
    prob_map = predict_sar_array(model, vv, vh, device, in_channels=in_channels)
    t_inf = time.time() - t0
    print(f"  U-Net Inference Time: {t_inf * 1000:.1f}ms")

    # --- 4. Permanent Water Masking ---
    perm_water_mask = None
    if args.pre_flood_s1:
        print(f"\nEvaluating Pre-Flood Baseline SAR: {args.pre_flood_s1}")
        with rasterio.open(args.pre_flood_s1) as p_ds:
            pre_vv = p_ds.read(1).astype(np.float32)
            pre_vh = p_ds.read(2).astype(np.float32)
        pre_prob = predict_sar_array(model, pre_vv, pre_vh, device, in_channels=in_channels)
        perm_water_mask = pre_prob > args.threshold
        print(f"  Identified permanent water baseline: {int(perm_water_mask.sum()):,} pixels")
    elif args.permanent_water:
        print(f"\nApplying Permanent Water Layer: {args.permanent_water}")
        perm_water_resampled = resample_raster_to_reference(args.permanent_water, s1_path)
        perm_water_mask = perm_water_resampled > 0.5
        print(f"  Permanent water pixels: {int(perm_water_mask.sum()):,}")

    # --- 5. DEM Slope Masking ---
    slope_deg = None
    if args.dem:
        print(f"\nProcessing DEM: {args.dem}")
        dem_resampled = resample_raster_to_reference(args.dem, s1_path)
        slope_deg = compute_terrain_slope(dem_resampled, pixel_size_x_m, pixel_size_y_m)
        print(f"  Terrain Slope: Min={slope_deg.min():.1f}°, Max={slope_deg.max():.1f}°, Mean={slope_deg.mean():.1f}°")
        print(f"  Steep slopes (> {args.max_slope}°): {int((slope_deg > args.max_slope).sum()):,} pixels")
    else:
        # Auto-fetch DEM elevation
        dem_fetched = fetch_dem_for_raster(s1_path)
        if dem_fetched is not None and not np.isnan(dem_fetched).all():
            slope_deg = compute_terrain_slope(dem_fetched, pixel_size_x_m, pixel_size_y_m)
            print(f"  Auto-fetched DEM elevation: Mean slope={slope_deg.mean():.1f}°, Steep (> {args.max_slope}°): {int((slope_deg > args.max_slope).sum()):,} pixels")

    # --- 6. Assemble Disaster Response Map ---
    disaster_map, stats = generate_disaster_response_map(
        prob_map=prob_map,
        slope_deg=slope_deg,
        max_slope=args.max_slope,
        perm_water_mask=perm_water_mask,
        threshold=args.threshold,
    )

    print("\nDisaster Response Summary:")
    print(f"  Active Inundated Flood:  {stats['flood_pixels']:,} pixels ({stats['flood_area_sqkm']:.2f} km²)")
    if stats["permanent_water_pixels"] > 0:
        print(f"  Permanent Water Bodies:  {stats['permanent_water_pixels']:,} pixels ({stats['permanent_water_area_sqkm']:.2f} km²)")
    if stats["slope_masked_pixels"] > 0:
        print(f"  Radar Shadow Excluded:   {stats['slope_masked_pixels']:,} pixels")

    # --- 7. Compare with Ground Truth (if available) ---
    has_gt = label_path is not None and os.path.exists(label_path)
    if has_gt:
        with rasterio.open(label_path) as ds:
            gt_lbl = ds.read(1)
        valid = gt_lbl >= 0
        gt_binary = gt_lbl == 1

        # Compare non-slope masked detections with GT
        pred_binary = ((disaster_map == 1) | (disaster_map == 2)) & (disaster_map != 255)
        v_pred = pred_binary[valid]
        v_gt = gt_binary[valid]

        tp = int((v_pred & v_gt).sum())
        fp = int((v_pred & (~v_gt)).sum())
        fn = int(((~v_pred) & v_gt).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        print(f"\nGround Truth Accuracy (Valid Pixels):")
        print(f"  IoU:        {iou * 100:.2f}%")
        print(f"  F1 Score:   {f1 * 100:.2f}%")
        print(f"  Precision:  {precision * 100:.2f}%")
        print(f"  Recall:     {recall * 100:.2f}%")

    # --- 8. Save Output GeoTIFFs ---
    map_path = os.path.join(args.output_dir, f"{chip_id}_disaster_map.tif")
    prob_path = os.path.join(args.output_dir, f"{chip_id}_flood_prob.tif")
    save_multiclass_geotiff(map_path, disaster_map, s1_path, nodata=255)
    save_multiclass_geotiff(prob_path, prob_map, s1_path, nodata=255)

    print(f"\nSaved Disaster Response GeoTIFFs:")
    print(f"  Disaster Map:    {map_path}")
    print(f"  Probability Map: {prob_path}")

    # --- 9. Generate Multi-Panel Disaster Visualization ---
    panels = []
    titles = []

    # Panel 1: SAR VV
    panels.append(("gray", vv, np.percentile(vv, 2), np.percentile(vv, 98), None))
    titles.append("Sentinel-1 SAR (VV)")

    # Panel 2: SAR VH
    panels.append(("gray", vh, np.percentile(vh, 2), np.percentile(vh, 98), None))
    titles.append("Sentinel-1 SAR (VH)")

    # Panel 3: Terrain Slope (if available)
    if slope_deg is not None:
        panels.append(("magma", slope_deg, 0, 15, "Slope (°)"))
        titles.append(f"Terrain Slope (°)\n(Mask: >{args.max_slope}°)")

    # Panel 4: Ground Truth (if available)
    if has_gt:
        gt_vis = np.zeros((*gt_lbl.shape, 3), dtype=np.float32)
        gt_vis[gt_lbl == 0] = [0.15, 0.15, 0.15]  # Dry
        gt_vis[gt_lbl == 1] = [0.1, 0.6, 1.0]     # Water
        gt_vis[gt_lbl == -1] = [0.4, 0.4, 0.4]    # Nodata/cloud
        panels.append(("rgb", gt_vis, None, None, None))
        titles.append("Ground Truth\n(Hand-Labeled)")

    # Panel 5: Raw U-Net Probability
    panels.append(("inferno", prob_map, 0.0, 1.0, "Prob"))
    titles.append("U-Net Flood\nProbability")

    # Panel 6: Final Operational Disaster Response Map
    map_vis = np.zeros((H, W, 3), dtype=np.float32)
    map_vis[disaster_map == 0] = [0.12, 0.12, 0.12]   # Dry land (dark)
    map_vis[disaster_map == 1] = [1.0, 0.15, 0.15]    # Active Flood Inundation (Red)
    map_vis[disaster_map == 2] = [0.1, 0.55, 1.0]     # Permanent Water (Blue)
    map_vis[disaster_map == 255] = [0.4, 0.35, 0.3]   # Steep Slope Excluded (Brown/Grey)
    panels.append(("rgb", map_vis, None, None, None))
    titles.append("Operational Disaster Map\n(Red=Flood, Blue=Perm Water)")

    num_panels = len(panels)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.2 * num_panels, 4.2))

    for idx, (cmap_type, data, vmin, vmax, cbar_label) in enumerate(panels):
        ax = axes[idx]
        if cmap_type == "rgb":
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap_type, vmin=vmin, vmax=vmax)
            if cbar_label:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{chip_id}_disaster_response.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Disaster Map Figure: {plot_path}")


if __name__ == "__main__":
    main()
