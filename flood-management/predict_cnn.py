#!/usr/bin/env python3
"""
Operational Disaster Response Flood Prediction & Mapping

Runs trained U-Net inference on Sentinel-1 SAR imagery with production-grade
post-processing:
  1. DEM Slope Masking: Eliminates mountain radar shadows (slope > 5.0 deg).
  2. Permanent Water Separation: Distinguishes permanent rivers/lakes from active flood inundation.
  3. Multi-Class Disaster Map Export & Visualization.

Works on a single 512x512 chip or a full Sentinel-1 scene: the raster is read,
predicted and written in BLOCK x BLOCK windows (with a MARGIN of context), and
each window is predicted in 512 px tiles - the chip size the model was trained
on - blended with a Hann window. Memory stays at a few hundred MB either way.

Usage:
    uv run python predict_cnn.py --model models/best_model.pth --chip-id USA_905409
    uv run python predict_cnn.py --model models/best_model.pth --input-s1 path/to/s1.tif --dem path/to/dem.tif
    uv run python predict_cnn.py --input-s1 scene.tif --bbox 92.6 26.4 93.1 26.8   # crop to an AOI (lon/lat)

The *_disaster_map.tif it writes is what ../Clustering/sar_scenes.py ingests to
build priority zones and evacuation routes.
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
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from cnn_model import get_model, UNet
from postprocess import (
    compute_terrain_slope,
    fetch_dem_on_grid,
    generate_disaster_response_map,
    resample_raster_to_grid,
)
from sen1floods11 import load_chip_ids_from_csv, normalize_s1_band
from test_custom_s1 import get_hann_window_2d, predict_patch_tta

TILE = 512            # model input size = Sen1Floods11 chip size
TILE_STRIDE = 384     # 128 px overlap between neighbouring tiles
TILE_BATCH = 4
BLOCK = 2048          # pixels per side read from disk at a time
MARGIN = 128          # context read around each block and then discarded
PREVIEW_MAX = 1600    # longest side of the arrays kept for the figure


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def resolve_model_path(path):
    """--model if given, else models/best_model.pth, else best_model.pth next to this script."""
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ("models/best_model.pth",
              os.path.join(here, "models", "best_model.pth"),
              os.path.join(here, "best_model.pth")):
        if os.path.exists(p):
            return p
    return "models/best_model.pth"


def load_model(path, device):
    """
    Model from a train_cnn.py checkpoint: best_model.pth (dict with arch,
    in_channels, metrics) or final_model.pth (bare state_dict - arch and input
    channels are read off the weights).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    meta = checkpoint if "model_state_dict" in checkpoint else {}
    arch = meta.get("arch") or ("resnet34" if "bottleneck.project.0.weight" in state else "unet")
    first_conv = state["conv1.weight"] if arch == "resnet34" else state["inc.conv.0.weight"]
    in_channels = meta.get("in_channels", int(first_conv.shape[1]))
    model = get_model(arch=arch, in_channels=in_channels, pretrained=False)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, arch, in_channels, meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _tile_starts(n):
    starts = list(range(0, n - TILE + 1, TILE_STRIDE))
    if starts[-1] + TILE < n:
        starts.append(n - TILE)
    return starts


def predict_sar_array(model, vv: np.ndarray, vh: np.ndarray, device, in_channels: int = 2,
                      valid: np.ndarray = None, use_tta: bool = False) -> np.ndarray:
    """
    Run model on VV and VH 2D arrays of any size, returning a 2D probability map.

    Arrays smaller than a tile are reflect-padded (as in training); larger ones
    are covered by overlapping 512 px tiles. Tiles with no valid pixel are skipped.
    """
    vv_norm = normalize_s1_band(vv)
    vh_norm = normalize_s1_band(vh)

    if in_channels == 3:
        diff_norm = np.clip((vh_norm - vv_norm + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        x = np.stack([vv_norm, vh_norm, diff_norm], axis=0)
    else:
        x = np.stack([vv_norm, vh_norm], axis=0)

    H, W = vv.shape
    ph, pw = max(H, TILE), max(W, TILE)
    if (ph, pw) != (H, W):
        x = np.pad(x, ((0, 0), (0, ph - H), (0, pw - W)), mode="reflect")
    if valid is not None and (ph, pw) != (H, W):
        valid = np.pad(valid, ((0, ph - H), (0, pw - W)), constant_values=False)

    tiles = [(ty, tx) for ty in _tile_starts(ph) for tx in _tile_starts(pw)
             if valid is None or valid[ty:ty + TILE, tx:tx + TILE].any()]
    window = get_hann_window_2d(TILE)
    prob_accum = np.zeros((ph, pw), dtype=np.float32)
    weight_accum = np.zeros((ph, pw), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), TILE_BATCH):
            batch = tiles[i:i + TILE_BATCH]
            tensor_in = torch.from_numpy(
                np.stack([x[:, ty:ty + TILE, tx:tx + TILE] for ty, tx in batch])).to(device)
            probs = predict_patch_tta(model, tensor_in, use_tta=use_tta)[:, 0].cpu().numpy()
            for (ty, tx), p in zip(batch, probs):
                prob_accum[ty:ty + TILE, tx:tx + TILE] += p * window
                weight_accum[ty:ty + TILE, tx:tx + TILE] += window

    prob = np.where(weight_accum > 0, prob_accum / np.maximum(weight_accum, 1e-6), 0.0)[:H, :W]
    if valid is not None:
        prob[~valid[:H, :W]] = 0.0
    return prob.astype(np.float32)


def valid_sar(vv: np.ndarray, vh: np.ndarray, nodata=None) -> np.ndarray:
    """Pixels that carry radar data: finite, not the nodata value, not zero-filled swath edge."""
    ok = np.isfinite(vv) & np.isfinite(vh) & ~((vv == 0) & (vh == 0))
    if nodata is not None and not np.isnan(nodata):
        ok &= (vv != nodata) & (vh != nodata)
    return ok


def pixel_size_m(transform, crs, center_lat):
    """Pixel size in metres (x, y), converting degrees if the CRS is geographic."""
    res_x, res_y = abs(transform.a), abs(transform.e)
    if crs and crs.is_geographic:
        return res_x * 111320.0 * math.cos(math.radians(center_lat)), res_y * 111320.0
    return (res_x if res_x > 0 else 10.0), (res_y if res_y > 0 else 10.0)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _out_profile(ds, window, dtype, nodata):
    return {
        "driver": "GTiff", "count": 1, "dtype": dtype, "nodata": nodata,
        "width": int(window.width), "height": int(window.height),
        "crs": ds.crs, "transform": ds.window_transform(window),
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


def _put_preview(preview, arr, r0, c0, f):
    """Write every f-th pixel of a block (at global row r0, col c0) into a decimated preview."""
    sy, sx = (-r0) % f, (-c0) % f
    sub = arr[sy::f, sx::f]
    py, px = (r0 + sy) // f, (c0 + sx) // f
    h = min(sub.shape[0], preview.shape[0] - py)
    w = min(sub.shape[1], preview.shape[1] - px)
    if h > 0 and w > 0:
        preview[py:py + h, px:px + w] = sub[:h, :w]


def main():
    parser = argparse.ArgumentParser(description="Operational U-Net Flood Detection with DEM & Permanent Water Masking")
    parser.add_argument("--model", default=None, help="Path to trained checkpoint (default: models/best_model.pth, else ./best_model.pth)")
    parser.add_argument("--data-dir", default="data/sen1floods11", help="Sen1Floods11 dataset directory")
    parser.add_argument("--chip-id", default=None, help="Chip ID to evaluate")
    parser.add_argument("--input-s1", default=None, help="Path to external 2-band S1 GeoTIFF (Band 1: VV, Band 2: VH)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"), default=None,
                        help="Only process this lon/lat box of the input (e.g. the town around a flood)")
    parser.add_argument("--dem", default=None, help="Path to Digital Elevation Model (DEM) GeoTIFF")
    parser.add_argument("--max-slope", type=float, default=5.0, help="Max slope angle in degrees to filter mountain radar shadows (default: 5.0)")
    parser.add_argument("--pre-flood-s1", default=None, help="Path to pre-flood Sentinel-1 SAR raster to detect permanent water baseline")
    parser.add_argument("--permanent-water", default=None, help="Path to permanent water mask raster (e.g. JRC GSW)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for surface water")
    parser.add_argument("--tta", action="store_true", help="4-fold test-time augmentation (slower, slightly smoother)")
    parser.add_argument("--output-dir", default="predictions", help="Output directory for predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("=" * 65)
    print("Operational Disaster Response Flood Mapping")
    print("=" * 65)
    print(f"Device: {device}")

    # --- 1. Load Model ---
    model_path = resolve_model_path(args.model)
    if not os.path.exists(model_path):
        print(f"[Error] Model checkpoint not found at '{model_path}'. Please train the model first.", file=sys.stderr)
        return

    model, arch, in_channels, meta = load_model(model_path, device)
    val_iou = meta.get("val_iou", 0.0)
    print(f"Loaded {arch.upper()} model from '{model_path}' (in_channels={in_channels}, Best Val IoU={val_iou*100:.2f}%)")

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

    map_path = os.path.join(args.output_dir, f"{chip_id}_disaster_map.tif")
    prob_path = os.path.join(args.output_dir, f"{chip_id}_flood_prob.tif")

    with rasterio.open(s1_path) as ds:
        has_crs = ds.crs is not None
        full = Window(0, 0, ds.width, ds.height)
        aoi = full
        if args.bbox:
            if not has_crs:
                print("[Error] --bbox needs a georeferenced input (this raster has no CRS).", file=sys.stderr)
                return
            b = args.bbox if ds.crs.to_epsg() == 4326 else transform_bounds("EPSG:4326", ds.crs, *args.bbox)
            aoi = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
            aoi = aoi.intersection(full)
            print(f"  AOI (--bbox):   {int(aoi.width)} x {int(aoi.height)} px of {ds.width} x {ds.height}")

        H, W = int(aoi.height), int(aoi.width)
        center_lat = 0.0
        if has_crs:
            _, south, _, north = transform_bounds(ds.crs, "EPSG:4326", *window_bounds(aoi, ds.transform))
            center_lat = (south + north) / 2.0
        pixel_size_x_m, pixel_size_y_m = pixel_size_m(ds.transform, ds.crs, center_lat)
        print(f"  Raster Size:    {W} x {H} pixels (Resolution: {pixel_size_x_m:.1f}m x {pixel_size_y_m:.1f}m)")

        f = max(1, math.ceil(max(H, W) / PREVIEW_MAX))
        pv = {k: np.full((math.ceil(H / f), math.ceil(W / f)), np.nan, np.float32)
              for k in ("vv", "vh", "slope", "prob", "map")}
        stats = {k: 0 for k in ("flood_pixels", "permanent_water_pixels", "slope_masked_pixels", "nodata_pixels")}
        stats.update(flood_area_sqkm=0.0, permanent_water_area_sqkm=0.0)
        have_dem = False
        n_blocks = math.ceil(H / BLOCK) * math.ceil(W / BLOCK)

        if args.pre_flood_s1:
            print(f"  Pre-Flood SAR:  {args.pre_flood_s1} (permanent water baseline)")
        elif args.permanent_water:
            print(f"  Permanent Water Layer: {args.permanent_water}")
        if args.dem:
            print(f"  DEM:            {args.dem}")

        # --- 3-6. Inference, permanent water, slope mask and map, block by block ---
        t0 = time.time()
        with rasterio.open(map_path, "w", **_out_profile(ds, aoi, "uint8", 255)) as map_ds, \
             rasterio.open(prob_path, "w", **_out_profile(ds, aoi, "float32", float("nan"))) as prob_ds:
            done = 0
            for r in range(0, H, BLOCK):
                for c in range(0, W, BLOCK):
                    bh, bw = min(BLOCK, H - r), min(BLOCK, W - c)
                    # block plus margin, in source pixel coordinates, clipped to the raster
                    r0 = max(int(aoi.row_off) + r - MARGIN, 0)
                    c0 = max(int(aoi.col_off) + c - MARGIN, 0)
                    r1 = min(int(aoi.row_off) + r + bh + MARGIN, ds.height)
                    c1 = min(int(aoi.col_off) + c + bw + MARGIN, ds.width)
                    win = Window(c0, r0, c1 - c0, r1 - r0)
                    iy = int(aoi.row_off) + r - r0
                    ix = int(aoi.col_off) + c - c0
                    inner = (slice(iy, iy + bh), slice(ix, ix + bw))
                    out_win = Window(c, r, bw, bh)
                    done += 1

                    vv = ds.read(1, window=win).astype(np.float32)
                    vh = ds.read(2, window=win).astype(np.float32)
                    valid = valid_sar(vv, vh, ds.nodata)
                    if not valid[inner].any():
                        map_ds.write(np.full((bh, bw), 255, np.uint8), 1, window=out_win)
                        prob_ds.write(np.full((bh, bw), np.nan, np.float32), 1, window=out_win)
                        stats["nodata_pixels"] += bh * bw
                        continue

                    prob = predict_sar_array(model, vv, vh, device, in_channels, valid, args.tta)
                    win_tr = ds.window_transform(win)
                    grid = (win_tr, vv.shape, ds.crs)

                    perm_water_mask = None
                    if args.pre_flood_s1:
                        pre_vv = resample_raster_to_grid(args.pre_flood_s1, *grid, band=1)
                        pre_vh = resample_raster_to_grid(args.pre_flood_s1, *grid, band=2)
                        pre_prob = predict_sar_array(model, pre_vv, pre_vh, device, in_channels,
                                                     valid_sar(pre_vv, pre_vh), args.tta)
                        perm_water_mask = pre_prob > args.threshold
                    elif args.permanent_water:
                        perm_water_mask = np.nan_to_num(resample_raster_to_grid(args.permanent_water, *grid)) > 0.5

                    slope_deg = None
                    if has_crs:
                        dem = (resample_raster_to_grid(args.dem, *grid) if args.dem
                               else fetch_dem_on_grid(*grid))
                        if dem is not None and not np.isnan(dem).all():
                            have_dem = True
                            slope_deg = compute_terrain_slope(dem, pixel_size_x_m, pixel_size_y_m)

                    disaster_map, _ = generate_disaster_response_map(
                        prob_map=prob,
                        slope_deg=slope_deg,
                        max_slope=args.max_slope,
                        perm_water_mask=perm_water_mask,
                        threshold=args.threshold,
                        valid_mask=valid,
                    )
                    block_map = disaster_map[inner]
                    block_prob = np.where(valid[inner], prob[inner], np.nan).astype(np.float32)
                    map_ds.write(block_map, 1, window=out_win)
                    prob_ds.write(block_prob, 1, window=out_win)

                    # statistics over the block proper, not its margin
                    blat = (win_tr * (0, iy + bh / 2))[1] if has_crs and ds.crs.is_geographic else center_lat
                    px_x, px_y = pixel_size_m(ds.transform, ds.crs, blat)
                    n_flood = int((block_map == 1).sum())
                    n_perm = int((block_map == 2).sum())
                    n_nodata = int((~valid[inner]).sum())
                    stats["flood_pixels"] += n_flood
                    stats["permanent_water_pixels"] += n_perm
                    stats["nodata_pixels"] += n_nodata
                    stats["slope_masked_pixels"] += int((block_map == 255).sum()) - n_nodata
                    stats["flood_area_sqkm"] += n_flood * px_x * px_y / 1e6
                    stats["permanent_water_area_sqkm"] += n_perm * px_x * px_y / 1e6

                    _put_preview(pv["vv"], np.where(valid, vv, np.nan)[inner], r, c, f)
                    _put_preview(pv["vh"], np.where(valid, vh, np.nan)[inner], r, c, f)
                    _put_preview(pv["prob"], block_prob, r, c, f)
                    _put_preview(pv["map"], block_map.astype(np.float32), r, c, f)
                    if slope_deg is not None:
                        _put_preview(pv["slope"], slope_deg[inner], r, c, f)
                    print(f"  Block {done}/{n_blocks}: {stats['flood_area_sqkm']:.2f} km² flood so far", end="\r")

        t_inf = time.time() - t0
        print(f"\n  U-Net Inference + Post-processing Time: {t_inf:.1f}s ({n_blocks} block(s))")
        if not have_dem:
            print("  [Note] No DEM available - steep-slope radar shadows were not masked.")

    print("\nDisaster Response Summary:")
    print(f"  Active Inundated Flood:  {stats['flood_pixels']:,} pixels ({stats['flood_area_sqkm']:.2f} km²)")
    if stats["permanent_water_pixels"] > 0:
        print(f"  Permanent Water Bodies:  {stats['permanent_water_pixels']:,} pixels ({stats['permanent_water_area_sqkm']:.2f} km²)")
    if stats["slope_masked_pixels"] > 0:
        print(f"  Radar Shadow Excluded:   {stats['slope_masked_pixels']:,} pixels")
    if stats["nodata_pixels"] > 0:
        print(f"  No SAR Data (excluded):  {stats['nodata_pixels']:,} pixels")

    # --- 7. Compare with Ground Truth (if available) ---
    has_gt = label_path is not None and os.path.exists(label_path)
    if has_gt:
        with rasterio.open(label_path) as lds:
            gt_lbl = lds.read(1)
        with rasterio.open(map_path) as mds:
            disaster_map = mds.read(1)
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

    # --- 8. Saved Output GeoTIFFs ---
    print(f"\nSaved Disaster Response GeoTIFFs:")
    print(f"  Disaster Map:    {map_path}")
    print(f"  Probability Map: {prob_path}")

    # --- 9. Generate Multi-Panel Disaster Visualization ---
    panels = []
    titles = []

    # Panels 1-2: SAR VV / VH
    vv, vh = pv["vv"], pv["vh"]
    if np.isfinite(vv).any():
        panels.append(("gray", vv, np.nanpercentile(vv, 2), np.nanpercentile(vv, 98), None))
        titles.append("Sentinel-1 SAR (VV)")
        panels.append(("gray", vh, np.nanpercentile(vh, 2), np.nanpercentile(vh, 98), None))
        titles.append("Sentinel-1 SAR (VH)")

    # Panel 3: Terrain Slope (if available)
    if have_dem:
        panels.append(("magma", pv["slope"], 0, 15, "Slope (°)"))
        titles.append(f"Terrain Slope (°)\n(Mask: >{args.max_slope}°)")

    # Panel 4: Ground Truth (if available)
    if has_gt and f == 1:
        gt_vis = np.zeros((*gt_lbl.shape, 3), dtype=np.float32)
        gt_vis[gt_lbl == 0] = [0.15, 0.15, 0.15]  # Dry
        gt_vis[gt_lbl == 1] = [0.1, 0.6, 1.0]     # Water
        gt_vis[gt_lbl == -1] = [0.4, 0.4, 0.4]    # Nodata/cloud
        panels.append(("rgb", gt_vis, None, None, None))
        titles.append("Ground Truth\n(Hand-Labeled)")

    # Panel 5: Raw U-Net Probability
    panels.append(("inferno", pv["prob"], 0.0, 1.0, "Prob"))
    titles.append("U-Net Flood\nProbability")

    # Panel 6: Final Operational Disaster Response Map
    dmap = pv["map"]
    map_vis = np.zeros((*dmap.shape, 3), dtype=np.float32)
    map_vis[dmap == 0] = [0.12, 0.12, 0.12]   # Dry land (dark)
    map_vis[dmap == 1] = [1.0, 0.15, 0.15]    # Active Flood Inundation (Red)
    map_vis[dmap == 2] = [0.1, 0.55, 1.0]     # Permanent Water (Blue)
    map_vis[(dmap == 255) | np.isnan(dmap)] = [0.4, 0.35, 0.3]   # Steep Slope / No Data (Brown/Grey)
    panels.append(("rgb", map_vis, None, None, None))
    titles.append("Operational Disaster Map\n(Red=Flood, Blue=Perm Water)")

    num_panels = len(panels)
    fig, axes = plt.subplots(1, num_panels, figsize=(4.2 * num_panels, 4.2))
    axes = np.atleast_1d(axes)

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

    if stats["flood_pixels"] > 0 and has_crs:
        print("\nNext - priority zones and evacuation routes from this map:")
        print(f"  cd ../Clustering && python sar_scenes.py ingest {os.path.abspath(map_path)} --name \"<place, date>\"")


if __name__ == "__main__":
    main()
