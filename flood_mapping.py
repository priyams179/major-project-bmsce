#!/usr/bin/env python3
"""
Sentinel-1 GRD Flood Mapping

Implements flood detection from Sentinel-1 GRD products using:
  1. Radiometric calibration (DN -> sigma nought dB)
  2. Thermal range noise removal
  3. Refined Lee speckle filtering (port of GEE implementation)
  4. Change detection via ratio of before/after in dB
  5. Threshold-based flood detection
  6. Isolated pixel (small component) removal

Based on the Google Earth Engine flood mapping code for the August 2018
Kerala floods.  The GEE code operates on the pre-calibrated COPERNICUS/S1_GRD
collection; this script works on raw Sentinel-1 GRD SAFE archives so it
performs calibration and noise removal locally.

Usage:
    .venv/bin/python flood_mapping.py [--diff-threshold 1.25] [--min-pixels 8]
"""

import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIXEL_SIZE = 10.0          # metres (range and azimuth for GRD)
NUM_BANDS = 1
NODATA = 0.0
VIS_DOWNSAMPLE = 10        # stride for visualization


# ---------------------------------------------------------------------------
# SAFE directory helpers
# ---------------------------------------------------------------------------
def find_safe_dirs(data_dir):
    """Return sorted list of .SAFE directory paths."""
    dirs = sorted(glob.glob(os.path.join(data_dir, "*.SAFE")))
    if len(dirs) < 2:
        raise FileNotFoundError(
            f"Need at least 2 SAFE directories in {data_dir}; found {len(dirs)}"
        )
    return dirs


def find_vh_tiff(safe_dir):
    """Return path to the VH measurement TIFF inside a SAFE directory."""
    matches = glob.glob(os.path.join(safe_dir, "measurement", "*vh*.tiff"))
    if not matches:
        raise FileNotFoundError(f"No VH TIFF found in {safe_dir}/measurement/")
    return matches[0]


def find_calibration_xml(safe_dir):
    """Return path to the VH calibration XML."""
    matches = glob.glob(
        os.path.join(safe_dir, "annotation", "calibration", "calibration-*vh*.xml")
    )
    if not matches:
        raise FileNotFoundError(f"No calibration XML for VH in {safe_dir}")
    return matches[0]


def find_noise_xml(safe_dir):
    """Return path to the VH noise XML."""
    matches = glob.glob(
        os.path.join(safe_dir, "annotation", "calibration", "noise-*vh*.xml")
    )
    if not matches:
        raise FileNotFoundError(f"No noise XML for VH in {safe_dir}")
    return matches[0]


def get_image_date(safe_dir):
    """Extract acquisition date (YYYYMMDD) from SAFE directory name."""
    basename = os.path.basename(safe_dir)
    # e.g. S1A_IW_GRDH_1SDV_20180814T004813_...
    return basename.split("_")[4][:8]


# ---------------------------------------------------------------------------
# XML parsing + LUT interpolation
# ---------------------------------------------------------------------------
def _parse_vector_list(root, list_tag, vector_tag, value_tag):
    """
    Parse a generic Sentinel-1 XML vector list.

    Returns (lines, pixels, values_2d) where lines is a 1-D array of azimuth
    positions, pixels is a 1-D array of range positions (shared across all
    vectors), and values_2d has shape (len(lines), len(pixels)).
    """
    vlist = root.find(f".//{list_tag}")
    if vlist is None:
        raise ValueError(f"<{list_tag}> not found in XML")

    count = int(vlist.get("count"))
    lines = np.empty(count, dtype=np.float64)
    pixels = None
    values = []

    for i, vec in enumerate(vlist):
        line_elem = vec.find("line")
        lines[i] = float(line_elem.text) if line_elem is not None else 0.0

        pix_elem = vec.find("pixel")
        pix_vals = np.array(
            [float(x) for x in pix_elem.text.split()], dtype=np.float64
        )
        if pixels is None:
            pixels = pix_vals
        elif len(pix_vals) != len(pixels):
            raise ValueError("Pixel count varies across vectors")

        val_elem = vec.find(value_tag)
        val_vals = np.array(
            [float(x) for x in val_elem.text.split()], dtype=np.float64
        )
        if len(val_vals) != len(pixels):
            raise ValueError(
                f"Value count ({len(val_vals)}) != pixel count ({len(pixels)})"
            )
        values.append(val_vals)

    return lines, pixels, np.array(values, dtype=np.float64)


def interpolate_lut(lines, pixels, values, n_lines, n_pixels):
    """
    Interpolate a 2-D LUT to full image size (n_lines x n_pixels).

    Uses linear interpolation with nearest-value extrapolation at the edges.
    """
    # Target axes
    target_lines = np.arange(n_lines, dtype=np.float64)
    target_pixels = np.arange(n_pixels, dtype=np.float64)

    # Build interpolator: values has shape (n_vectors, n_pixels_per_vector)
    interp = RegularGridInterpolator(
        (lines, pixels), values, method="linear", bounds_error=False,
        fill_value=None  # extrapolate with nearest
    )
    # Create meshgrid and interpolate
    LL, PP = np.meshgrid(target_lines, target_pixels, indexing="ij")
    points = np.column_stack([LL.ravel(), PP.ravel()])
    result = interp(points).reshape(n_lines, n_pixels)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration + noise removal
# ---------------------------------------------------------------------------
def calibrate_to_sigma_db(dn, cal_lut, noise_lut=None):
    """
    Calibrate raw DN values to sigma nought in dB.

    Parameters
    ----------
    dn : np.ndarray (uint16)
        Raw digital numbers from the measurement TIFF.
    cal_lut : np.ndarray (float32)
        Sigma nought calibration LUT interpolated to image size.
    noise_lut : np.ndarray (float32) or None
        Range noise LUT interpolated to image size (same units as DN^2).

    Returns
    -------
    sigma_db : np.ndarray (float32)
        Calibrated, denoised sigma nought in dB.
    """
    dn_f = dn.astype(np.float64)
    dn_sq = dn_f * dn_f

    if noise_lut is not None:
        dn_sq = np.maximum(dn_sq - noise_lut.astype(np.float64), 1.0)

    sigma_linear = dn_sq / (cal_lut.astype(np.float64) ** 2)
    sigma_linear = np.maximum(sigma_linear, 1e-10)  # avoid log(0)
    sigma_db = 10.0 * np.log10(sigma_linear)
    return sigma_db.astype(np.float32)


# ---------------------------------------------------------------------------
# Unit conversion helpers (matching GEE code)
# ---------------------------------------------------------------------------
def to_natural(img_db):
    """Convert dB to natural (linear) units."""
    return np.power(10.0, img_db.astype(np.float64) / 10.0)


def to_db(img_natural):
    """Convert natural (linear) units to dB."""
    return 10.0 * np.log10(np.maximum(img_natural, 1e-10))


# ---------------------------------------------------------------------------
# Refined Lee speckle filter
# ---------------------------------------------------------------------------
def _shift_array(arr, dy, dx):
    """Shift a 2-D array by integer (dy, dx) with reflect-padded boundaries."""
    pad_y = abs(dy)
    pad_x = abs(dx)
    if pad_y == 0 and pad_x == 0:
        return arr.copy()
    padded = np.pad(
        arr,
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="reflect",
    )
    y0 = pad_y + dy
    x0 = pad_x + dx
    return padded[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]].copy()


def refined_lee(img_natural):
    """
    Apply the Refined Lee speckle filter.

    Direct port of the Google Earth Engine implementation by Guido Lemoine,
    based on the SNAP 3.0 S1TBX RefinedLee filter.

    Parameters
    ----------
    img_natural : np.ndarray (float32/64)
        Image in natural (linear) units, NOT dB.

    Returns
    -------
    filtered : np.ndarray
        Filtered image in natural (linear) units.
    """
    img = img_natural.astype(np.float64)
    H, W = img.shape

    # --- Step 1: 3x3 mean and variance ---
    mean3 = ndimage.uniform_filter(img, size=3, mode="reflect")
    mean_sq3 = ndimage.uniform_filter(img * img, size=3, mode="reflect")
    variance3 = mean_sq3 - mean3 * mean3

    # --- Step 2: Sample at 9 positions in a 7x7 window ---
    # The sample kernel picks positions at offsets:
    #   (-2,-2) (-2, 0) (-2, 2)
    #   ( 0,-2) ( 0, 0) ( 0, 2)
    #   ( 2,-2) ( 2, 0) ( 2, 2)
    offsets = [
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2), (0, 0), (0, 2),
        (2, -2), (2, 0), (2, 2),
    ]
    sample_mean = np.empty((9, H, W), dtype=np.float32)
    sample_var = np.empty((9, H, W), dtype=np.float32)
    for i, (dy, dx) in enumerate(offsets):
        sample_mean[i] = _shift_array(mean3, dy, dx)
        sample_var[i] = _shift_array(variance3, dy, dx)

    del mean3, mean_sq3, variance3

    # --- Step 3: Compute 4 gradients ---
    # gradient 0: vertical   |sample_mean[1] - sample_mean[7]|
    # gradient 1: diagonal   |sample_mean[6] - sample_mean[2]|
    # gradient 2: horizontal |sample_mean[3] - sample_mean[5]|
    # gradient 3: anti-diag  |sample_mean[0] - sample_mean[8]|
    gradients = np.empty((4, H, W), dtype=np.float32)
    gradients[0] = np.abs(sample_mean[1] - sample_mean[7])
    gradients[1] = np.abs(sample_mean[6] - sample_mean[2])
    gradients[2] = np.abs(sample_mean[3] - sample_mean[5])
    gradients[3] = np.abs(sample_mean[0] - sample_mean[8])

    max_gradient = np.max(gradients, axis=0)

    # gradmask: which gradient is the maximum (duplicated for 8 directions)
    gradmask = np.empty((8, H, W), dtype=bool)
    for i in range(4):
        gradmask[i] = gradients[i] == max_gradient
        gradmask[i + 4] = gradmask[i]

    del gradients, max_gradient

    # --- Step 4: Determine 8 directions ---
    # For each of 4 gradient tests, decide which half of the direction pair
    # is active.  Directions 5-8 are the negation of directions 1-4.
    test1 = (sample_mean[1] - sample_mean[4]) > (sample_mean[4] - sample_mean[7])
    test2 = (sample_mean[6] - sample_mean[4]) > (sample_mean[4] - sample_mean[2])
    test3 = (sample_mean[3] - sample_mean[4]) > (sample_mean[4] - sample_mean[5])
    test4 = (sample_mean[0] - sample_mean[4]) > (sample_mean[4] - sample_mean[8])

    dir_vals = np.zeros((8, H, W), dtype=np.float32)
    dir_vals[0] = np.where(test1, 1, 0)
    dir_vals[1] = np.where(test2, 2, 0)
    dir_vals[2] = np.where(test3, 3, 0)
    dir_vals[3] = np.where(test4, 4, 0)
    dir_vals[4] = np.where(~test1, 5, 0)
    dir_vals[5] = np.where(~test2, 6, 0)
    dir_vals[6] = np.where(~test3, 7, 0)
    dir_vals[7] = np.where(~test4, 8, 0)

    # Apply gradmask (only keep the direction where this gradient is max)
    for i in range(8):
        np.putmask(dir_vals[i], ~gradmask[i], 0)

    # Collapse into single-band direction image
    directions = dir_vals.sum(axis=0)

    del gradmask, dir_vals, test1, test2, test3, test4

    # --- Step 5: Compute sigmaV (local noise variance) ---
    # sigmaV = mean of the 5 smallest values of (sample_var / sample_mean^2)
    safe_mean = np.maximum(sample_mean, 1e-10)
    sample_stats = sample_var / (safe_mean * safe_mean)
    sample_stats_sorted = np.sort(sample_stats, axis=0)
    sigmaV = sample_stats_sorted[:5].mean(axis=0).astype(np.float64)

    del sample_mean, sample_var, safe_mean, sample_stats, sample_stats_sorted

    # --- Step 6: Directional statistics with 7x7 kernels ---
    # Rectangular kernel: bottom 4 rows = 1, top 3 rows = 0
    rect = np.zeros((7, 7))
    rect[3:] = 1.0
    # Diagonal kernel: lower triangular
    diag = np.tril(np.ones((7, 7)))

    # GEE rotates clockwise; np.rot90(arr, -i) = clockwise by 90*i degrees
    kernels = [
        rect,                    # direction 1
        diag,                    # direction 2
        np.rot90(rect, -1),      # direction 3
        np.rot90(diag, -1),      # direction 4
        np.rot90(rect, -2),      # direction 5
        np.rot90(diag, -2),      # direction 6
        np.rot90(rect, -3),      # direction 7
        np.rot90(diag, -3),      # direction 8
    ]

    result = np.zeros_like(img)
    for i in range(8):
        direction_num = i + 1
        kernel = kernels[i]
        ksum = kernel.sum()

        dir_mean = ndimage.convolve(img, kernel, mode="reflect") / ksum
        dir_var = (
            ndimage.convolve(img * img, kernel, mode="reflect") / ksum
            - dir_mean * dir_mean
        )

        # Refined Lee formula
        # varX = (dir_var - dir_mean^2 * sigmaV) / (sigmaV + 1)
        # b = varX / dir_var
        # filtered = dir_mean + b * (img - dir_mean)
        varX = (dir_var - dir_mean * dir_mean * sigmaV) / (sigmaV + 1.0)
        b = np.where(dir_var > 1e-10, varX / dir_var, 0.0)
        filtered = dir_mean + b * (img - dir_mean)

        mask = directions == direction_num
        result = np.where(mask, filtered, result)

        del dir_mean, dir_var, varX, b, filtered, mask

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Simple Lee filter (faster alternative)
# ---------------------------------------------------------------------------
def lee_filter(img_natural, window=7):
    """
    Standard Lee sigma filter.

    Parameters
    ----------
    img_natural : np.ndarray
        Image in natural (linear) units.
    window : int
        Square filter window size (default 7).

    Returns
    -------
    filtered : np.ndarray
        Filtered image in natural (linear) units.
    """
    img = img_natural.astype(np.float64)
    mean = ndimage.uniform_filter(img, size=window, mode="reflect")
    mean_sq = ndimage.uniform_filter(img * img, size=window, mode="reflect")
    variance = mean_sq - mean * mean

    # Equivalent look count (assuming 1-look SAR)
    cu = np.sqrt(variance) / np.maximum(mean, 1e-10)
    ci = cu
    # Lee filter weight
    b = np.where(ci > 0, (1.0 - ci) / (1.0 + ci * ci), 0.0)
    # Actually the standard Lee filter uses:
    # b = variance / (variance + sigma_v^2)
    # where sigma_v^2 is the noise variance.  For 1-look, sigma_v^2 = mean^2.
    sigma_v_sq = mean * mean  # approximation for 1-look
    b = variance / np.maximum(variance + sigma_v_sq, 1e-10)

    filtered = mean + b * (img - mean)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def remove_small_components(mask, min_size=8, connectivity=8):
    """
    Remove connected components smaller than min_size.

    This replaces the GEE connectedPixelCount approach with global
    connected-component labelling, which is more effective for removing
    isolated flood pixels.
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    labeled, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask.copy()

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_size
    keep[0] = False  # background
    return keep[labeled]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_geotiff(path, data, dtype="float32"):
    """Save a 2-D array as a GeoTIFF with 10 m pixel size."""
    if dtype == "float32":
        gdtype = rasterio.float32
        nodata = float("nan")
    elif dtype == "uint8":
        gdtype = rasterio.uint8
        nodata = 0
    else:
        gdtype = rasterio.float32
        nodata = float("nan")

    h, w = data.shape
    transform = from_origin(0, h * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=gdtype,
        crs=None,  # raw GRD coordinates; geocoding required for map overlay
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype if dtype != "float32" else np.float32), 1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def create_visualization(
    before_db, after_db, before_filt, after_filt,
    flood_initial, flood_final, output_path,
):
    """Create a multi-panel figure and save as PNG."""
    ds = VIS_DOWNSAMPLE
    titles = [
        "Before Flood (dB)",
        "After Flood (dB)",
        "Before Filtered (dB)",
        "After Filtered (dB)",
        "Initial Flood Detection",
        "Final Flood Map",
    ]
    images = [before_db, after_db, before_filt, after_filt,
              flood_initial, flood_final]
    cmaps = ["gray", "gray", "gray", "gray",
             "Oranges", "Reds"]
    vmin_max = [(-25, 0), (-25, 0), (-25, 0), (-25, 0), (0, 1), (0, 1)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, title, img, cmap, (vmin, vmax) in zip(
        axes.ravel(), titles, images, cmaps, vmin_max
    ):
        sub = img[::ds, ::ds]
        ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.suptitle(
        "Sentinel-1 GRD Flood Mapping - Kerala Floods August 2018",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Visualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def print_statistics(flood_mask, pixel_size=PIXEL_SIZE):
    """Print flood statistics."""
    n_flooded = int(flood_mask.sum())
    pixel_area_ha = (pixel_size * pixel_size) / 1e4  # hectares per pixel
    flooded_area_ha = n_flooded * pixel_area_ha
    total_area_ha = flood_mask.size * pixel_area_ha
    pct = 100.0 * n_flooded / flood_mask.size if flood_mask.size > 0 else 0

    print(f"\n{'='*50}")
    print(f"Flood Mapping Statistics")
    print(f"{'='*50}")
    print(f"  Image size:        {flood_mask.shape[1]} x {flood_mask.shape[0]} pixels")
    print(f"  Pixel size:        {pixel_size:.0f} m x {pixel_size:.0f} m")
    print(f"  Total area:        {total_area_ha:,.1f} ha ({total_area_ha/100:.1f} km2)")
    print(f"  Flooded pixels:    {n_flooded:,}")
    print(f"  Flooded area:      {flooded_area_ha:,.1f} ha ({flooded_area_ha/100:.1f} km2)")
    print(f"  Flood coverage:    {pct:.2f}%")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def process_image(safe_dir, output_dir, label, apply_noise_removal=True):
    """
    Full processing pipeline for one Sentinel-1 GRD image:
    read -> calibrate -> denoise -> (return dB image)
    """
    t0 = time.time()
    vh_tiff = find_vh_tiff(safe_dir)
    cal_xml = find_calibration_xml(safe_dir)
    noise_xml = find_noise_xml(safe_dir)
    date_str = get_image_date(safe_dir)

    print(f"\n[{label}] Processing {date_str}")
    print(f"  TIFF:  {os.path.basename(vh_tiff)}")

    # --- Read TIFF ---
    with rasterio.open(vh_tiff) as ds:
        dn = ds.read(1)
        H, W = dn.shape
    print(f"  Size:  {W} x {H}")

    # --- Parse calibration LUT ---
    cal_tree = ET.parse(cal_xml)
    cal_root = cal_tree.getroot()
    cal_lines, cal_pixels, cal_values = _parse_vector_list(
        cal_root, "calibrationVectorList", "calibrationVector", "sigmaNought"
    )
    cal_lut = interpolate_lut(cal_lines, cal_pixels, cal_values, H, W)
    print(f"  Calibration LUT: {cal_lut.shape}, range [{cal_lut.min():.1f}, {cal_lut.max():.1f}]")

    # --- Parse noise LUT ---
    noise_lut = None
    if apply_noise_removal:
        noise_tree = ET.parse(noise_xml)
        noise_root = noise_tree.getroot()
        noise_lines, noise_pixels, noise_values = _parse_vector_list(
            noise_root, "noiseRangeVectorList", "noiseRangeVector", "noiseRangeLut"
        )
        noise_lut = interpolate_lut(noise_lines, noise_pixels, noise_values, H, W)
        print(f"  Noise LUT: {noise_lut.shape}, range [{noise_lut.min():.1f}, {noise_lut.max():.1f}]")

    # --- Calibrate ---
    sigma_db = calibrate_to_sigma_db(dn, cal_lut, noise_lut)
    print(f"  Sigma dB: range [{sigma_db.min():.2f}, {sigma_db.max():.2f}], "
          f"mean {sigma_db.mean():.2f}")

    # --- Save calibrated dB image ---
    out_path = os.path.join(output_dir, f"{label}_dB.tif")
    save_geotiff(out_path, sigma_db)
    print(f"  Saved: {out_path}")

    # --- Apply speckle filter ---
    print(f"  Applying Refined Lee speckle filter...")
    t1 = time.time()
    img_natural = to_natural(sigma_db)
    del sigma_db
    filtered_natural = refined_lee(img_natural)
    del img_natural
    filtered_db = to_db(filtered_natural).astype(np.float32)
    del filtered_natural
    print(f"  Filtered dB: range [{filtered_db.min():.2f}, {filtered_db.max():.2f}], "
          f"mean {filtered_db.mean():.2f}")
    print(f"  Speckle filter took {time.time() - t1:.1f}s")

    # --- Save filtered dB image ---
    filt_path = os.path.join(output_dir, f"{label}_filtered_dB.tif")
    save_geotiff(filt_path, filtered_db)
    print(f"  Saved: {filt_path}")
    print(f"  Total [{label}] time: {time.time() - t0:.1f}s")

    return filtered_db


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-1 GRD Flood Mapping for Kerala Floods 2018"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing extracted SAFE folders (default: data)",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=1.25,
        help="Ratio threshold for flood detection (default: 1.25, matches GEE)",
    )
    parser.add_argument(
        "--min-pixels", type=int, default=8,
        help="Minimum connected pixels to keep (default: 8, matches GEE)",
    )
    parser.add_argument(
        "--no-noise-removal", action="store_true",
        help="Skip thermal noise removal (faster, less accurate)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Find SAFE directories ---
    safe_dirs = find_safe_dirs(args.data_dir)
    before_dir = safe_dirs[0]  # earlier date
    after_dir = safe_dirs[1]   # later date
    print(f"Before flood: {get_image_date(before_dir)}")
    print(f"After flood:  {get_image_date(after_dir)}")

    # --- Process both images ---
    before_filt = process_image(
        before_dir, args.output_dir, "before",
        apply_noise_removal=not args.no_noise_removal,
    )
    after_filt = process_image(
        after_dir, args.output_dir, "after",
        apply_noise_removal=not args.no_noise_removal,
    )

    # --- Change detection: ratio of dB values ---
    # GEE code: difference = afterFiltered.divide(beforeFiltered)
    # flooded = difference.gt(diffThreshold)
    #
    # Both dB values are typically negative.  When flooding occurs, the
    # after-flood backscatter drops (more negative), so the ratio of two
    # negative numbers exceeds 1, crossing the threshold.
    print(f"\n{'='*50}")
    print("Change detection")
    print(f"{'='*50}")
    eps = 1e-6
    before_safe = np.where(np.abs(before_filt) > eps, before_filt, eps)
    difference = after_filt / before_safe
    print(f"  Ratio range: [{difference.min():.4f}, {difference.max():.4f}]")
    print(f"  Threshold:   {args.diff_threshold}")

    flood_initial = difference > args.diff_threshold
    print(f"  Initial flooded pixels: {int(flood_initial.sum()):,}")

    # --- Remove isolated pixels ---
    flood_final = remove_small_components(
        flood_initial, min_size=args.min_pixels, connectivity=8
    )
    print(f"  Final flooded pixels:    {int(flood_final.sum()):,}")
    print(f"  Removed isolated:        {int(flood_initial.sum() - flood_final.sum()):,}")

    # --- Save flood map ---
    flood_path = os.path.join(args.output_dir, "flood_map.tif")
    save_geotiff(flood_path, flood_final.astype(np.float32))
    print(f"\n  Flood map saved: {flood_path}")

    # --- Save ratio image ---
    ratio_path = os.path.join(args.output_dir, "change_ratio.tif")
    save_geotiff(ratio_path, difference.astype(np.float32))
    print(f"  Ratio image saved: {ratio_path}")

    # --- Visualization ---
    print(f"\nGenerating visualization...")
    # Re-read the calibrated dB images for visualization
    # (we only saved filtered versions, so use those)
    create_visualization(
        before_filt, after_filt, before_filt, after_filt,
        flood_initial, flood_final,
        os.path.join(args.output_dir, "flood_map_visualization.png"),
    )

    # --- Statistics ---
    print_statistics(flood_final)

    print("\nDone! All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
