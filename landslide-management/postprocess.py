#!/usr/bin/env python3
"""
Operational Landslide Detection Post-Processing Module

Provides:
  1. Terrain Slope Range Filtering: Excludes slopes < 5° (flat land where landslides
     do not initiate) and > 70° (sheer vertical cliffs / radar shadow artifacts).
     In India, >80% of landslides occur between 20° and 45°.
  2. Connected Component Filtering: Eliminates isolated noise specks (< 10 pixels).
  3. Pre-Existing Bare Rock / Soil Separation: Differentiates newly triggered
     landslides (Class 1) from pre-existing bare rock/soil outcrops (Class 2).
  4. Multi-Class Landslide Response Map Generation:
       0   : Stable Terrain
       1   : Newly Triggered Landslide Scar
       2   : Pre-Existing Bare Rock / Quarry / Permanent Unvegetated
       255 : Excluded (Gentle Slopes < 5°, Sheer Cliffs > 70°, Cloud/Sensor Nodata)
"""

import argparse
import math
import os
import sys
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Terrain Slope Calculation (Horn's Algorithm)
# ---------------------------------------------------------------------------
def compute_terrain_slope(
    dem: np.ndarray,
    pixel_size_x_m: float = 10.0,
    pixel_size_y_m: float = 10.0,
) -> np.ndarray:
    """
    Compute terrain slope in degrees using Horn's algorithm.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation array in metres.
    pixel_size_x_m, pixel_size_y_m : float
        Pixel spacing in metres.

    Returns
    -------
    slope_deg : np.ndarray (H, W)
        Slope in degrees [0, 90].
    """
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]], dtype=np.float32
    ) / (8.0 * max(pixel_size_x_m, 1e-3))

    kernel_y = np.array(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]], dtype=np.float32
    ) / (8.0 * max(pixel_size_y_m, 1e-3))

    dem_clean = np.nan_to_num(dem.astype(np.float32), nan=0.0)

    dz_dx = ndimage.convolve(dem_clean, kernel_x, mode="reflect")
    dz_dy = ndimage.convolve(dem_clean, kernel_y, mode="reflect")

    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.arctan(rise_run) * (180.0 / np.pi)
    return slope_deg.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Resample / Match Raster to Reference
# ---------------------------------------------------------------------------
def resample_raster_to_reference(src_path: str, ref_path: str) -> np.ndarray:
    """
    Reproject and resample a source raster to match dimensions and CRS of a reference raster.
    """
    with rasterio.open(ref_path) as ref_ds:
        dst_crs = ref_ds.crs
        dst_transform = ref_ds.transform
        dst_width = ref_ds.width
        dst_height = ref_ds.height

    destination = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src_ds:
        reproject(
            source=rasterio.band(src_ds, 1),
            destination=destination,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    return destination


# ---------------------------------------------------------------------------
# 3. DEM Auto-Fetcher (AWS Open Data Terrain Tiles)
# ---------------------------------------------------------------------------
def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to Web Mercator tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_dem_for_raster(ref_path: str, cache_dir: str = "data/dem_cache") -> Optional[np.ndarray]:
    """
    Fetch and reproject elevation data for a reference raster from AWS terrain tiles.
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with rasterio.open(ref_path) as ds:
            dst_crs = ds.crs
            dst_transform = ds.transform
            dst_w = ds.width
            dst_h = ds.height
            bounds = ds.bounds

            if dst_crs and not dst_crs.is_geographic:
                wgs84_bounds = transform_bounds(dst_crs, "EPSG:4326", *bounds)
            else:
                wgs84_bounds = bounds

        min_x, min_y, max_x, max_y = wgs84_bounds
        center_lat = (min_y + max_y) / 2.0
        center_lon = (min_x + max_x) / 2.0

        zoom = 10
        xtile, ytile = deg2num(center_lat, center_lon, zoom)
        tile_url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{xtile}/{ytile}.tif"
        cached_tile = os.path.join(cache_dir, f"dem_{zoom}_{xtile}_{ytile}.tif")

        if not os.path.exists(cached_tile) or os.path.getsize(cached_tile) == 0:
            req = urllib.request.Request(tile_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(cached_tile + ".tmp", "wb") as out:
                out.write(resp.read())
            os.replace(cached_tile + ".tmp", cached_tile)

        destination = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
        with rasterio.open(cached_tile) as src_ds:
            reproject(
                source=rasterio.band(src_ds, 1),
                destination=destination,
                src_transform=src_ds.transform,
                src_crs=src_ds.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs if dst_crs else "EPSG:4326",
                resampling=Resampling.bilinear,
            )
        return destination
    except Exception as e:
        print(f"Warning: Could not auto-fetch DEM for {ref_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# 4. Landslide Specific Post-Processing Filters
# ---------------------------------------------------------------------------
def apply_slope_range_filter(
    prediction: np.ndarray,
    slope_deg: np.ndarray,
    min_slope: float = 5.0,
    max_slope: float = 70.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exclude predictions falling outside plausible landslide slope ranges.

    Parameters
    ----------
    prediction : np.ndarray (H, W) bool or binary
    slope_deg : np.ndarray (H, W) float32
    min_slope : float
        Slopes < min_slope (e.g. <5°) are flat plains, rarely initiating slides.
    max_slope : float
        Slopes > max_slope (e.g. >70°) are near-vertical cliffs / SAR radar shadows.

    Returns
    -------
    filtered_prediction : np.ndarray (H, W) bool
    excluded_mask : np.ndarray (H, W) bool
    """
    valid_slope = (slope_deg >= min_slope) & (slope_deg <= max_slope)
    excluded_mask = ~valid_slope
    filtered = prediction & valid_slope
    return filtered, excluded_mask


def remove_small_components(binary_map: np.ndarray, min_pixels: int = 10) -> np.ndarray:
    """
    Remove isolated connected component clusters smaller than min_pixels using 8-connectivity.
    """
    if min_pixels <= 1:
        return binary_map.copy()

    structure = np.ones((3, 3), dtype=bool)
    labeled, num_features = ndimage.label(binary_map > 0, structure=structure)
    if num_features == 0:
        return np.zeros_like(binary_map)

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False  # Background
    return keep[labeled].astype(np.uint8)


def apply_morphological_smoothing(
    binary_map: np.ndarray,
    operation: str = "opening",
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply binary opening (remove salt noise) or closing (bridge scarp cracks).
    """
    structure = ndimage.generate_binary_structure(2, 2)
    if kernel_size > 3:
        structure = ndimage.iterate_structure(structure, kernel_size // 2)

    if operation == "opening":
        smoothed = ndimage.binary_opening(binary_map > 0, structure=structure)
    elif operation == "closing":
        smoothed = ndimage.binary_closing(binary_map > 0, structure=structure)
    else:
        smoothed = binary_map > 0

    return smoothed.astype(np.uint8)


def separate_preexisting_bare(
    candidate_landslides: np.ndarray,
    pre_ndvi: Optional[np.ndarray] = None,
    pre_bsi: Optional[np.ndarray] = None,
    ndvi_threshold: float = 0.20,
    bsi_threshold: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate new landslide scars from pre-existing bare rocks, quarries, or river gravel.

    Parameters
    ----------
    candidate_landslides : np.ndarray bool
    pre_ndvi : np.ndarray float or None
        Pre-event Normalized Difference Vegetation Index.
    pre_bsi : np.ndarray float or None
        Pre-event Bare Soil Index.

    Returns
    -------
    is_new_landslide : np.ndarray bool
    is_preexisting : np.ndarray bool
    """
    if pre_ndvi is None and pre_bsi is None:
        return candidate_landslides, np.zeros_like(candidate_landslides, dtype=bool)

    pre_bare = np.zeros_like(candidate_landslides, dtype=bool)
    if pre_ndvi is not None:
        pre_bare |= (pre_ndvi < ndvi_threshold)
    if pre_bsi is not None:
        pre_bare |= (pre_bsi > bsi_threshold)

    is_preexisting = candidate_landslides & pre_bare
    is_new_landslide = candidate_landslides & (~is_preexisting)
    return is_new_landslide, is_preexisting


# ---------------------------------------------------------------------------
# 5. Multi-Class Landslide Response Map Assembler
# ---------------------------------------------------------------------------
def generate_landslide_response_map(
    prob_map: np.ndarray,
    slope_deg: Optional[np.ndarray] = None,
    pre_ndvi: Optional[np.ndarray] = None,
    pre_bsi: Optional[np.ndarray] = None,
    cloud_mask: Optional[np.ndarray] = None,
    threshold: float = 0.50,
    min_slope: float = 5.0,
    max_slope: float = 70.0,
    min_component_size: int = 10,
    smooth_operation: Optional[str] = "opening",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Assemble operational 4-class landslide response map.

    Classes:
      0   : Stable Terrain
      1   : Newly Triggered Landslide Scar
      2   : Pre-Existing Bare Rock / Quarry / River Gravel
      255 : Excluded (Slope out of bounds <5° or >70°, or Cloud/Nodata)

    Returns
    -------
    response_map : np.ndarray (H, W), dtype uint8
    stats : dict
    """
    H, W = prob_map.shape
    raw_candidates = prob_map >= threshold

    # 1. Connected component filtering
    cleaned = remove_small_components(raw_candidates, min_pixels=min_component_size)

    # 2. Morphological smoothing
    if smooth_operation:
        cleaned = apply_morphological_smoothing(cleaned, operation=smooth_operation, kernel_size=3)

    # 3. Pre-existing bare rock separation
    is_new, is_preexisting = separate_preexisting_bare(
        cleaned > 0, pre_ndvi=pre_ndvi, pre_bsi=pre_bsi
    )

    # 4. Construct base map
    response_map = np.zeros((H, W), dtype=np.uint8)
    response_map[is_preexisting] = 2
    response_map[is_new] = 1

    # 5. Slope filtering
    slope_excluded = 0
    if slope_deg is not None:
        if slope_deg.shape != (H, W):
            sh, sw = slope_deg.shape
            aligned = np.zeros((H, W), dtype=np.float32)
            aligned[:min(H, sh), :min(W, sw)] = slope_deg[:min(H, sh), :min(W, sw)]
            slope_deg = aligned

        out_of_bounds = (slope_deg < min_slope) | (slope_deg > max_slope)
        response_map[out_of_bounds] = 255
        slope_excluded = int(out_of_bounds.sum())

    # 6. Cloud / invalid mask
    cloud_excluded = 0
    if cloud_mask is not None:
        if cloud_mask.shape != (H, W):
            ch, cw = cloud_mask.shape
            aligned_cloud = np.ones((H, W), dtype=bool)
            aligned_cloud[:min(H, ch), :min(W, cw)] = cloud_mask[:min(H, ch), :min(W, cw)] > 0.5
            cloud_mask = aligned_cloud

        invalid_pixels = (cloud_mask <= 0.5)
        response_map[invalid_pixels] = 255
        cloud_excluded = int(invalid_pixels.sum())

    # Statistics
    valid_mask = response_map != 255
    new_landslide_px = int((response_map == 1).sum())
    preexisting_px = int((response_map == 2).sum())
    stable_px = int((response_map == 0).sum())

    stats = {
        "new_landslide_pixels": new_landslide_px,
        "new_landslide_area_sqkm": (new_landslide_px * 100.0) / 1e6,  # 10m pixels
        "preexisting_bare_pixels": preexisting_px,
        "preexisting_bare_area_sqkm": (preexisting_px * 100.0) / 1e6,
        "stable_pixels": stable_px,
        "slope_excluded_pixels": slope_excluded,
        "cloud_excluded_pixels": cloud_excluded,
        "total_valid_pixels": int(valid_mask.sum()),
    }

    return response_map, stats


# ---------------------------------------------------------------------------
# 6. Save GeoTIFF
# ---------------------------------------------------------------------------
def save_landslide_map(
    response_map: np.ndarray,
    ref_profile: dict,
    output_path: str,
) -> None:
    """Save multi-class landslide map as GeoTIFF with colormap and metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    profile = ref_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 255,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(response_map.astype(np.uint8), 1)
        # 0: Green (Stable), 1: Red (Landslide), 2: Orange (Bare), 255: Transparent/Gray
        colormap = {
            0: (34, 139, 34, 255),    # Forest green
            1: (220, 20, 60, 255),    # Crimson red
            2: (255, 140, 0, 255),    # Dark orange
            255: (128, 128, 128, 0),  # Transparent nodata
        }
        dst.write_colormap(1, colormap)

    print(f"  Saved landslide response map to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Operational Post-Processing for Landslide Maps")
    parser.add_argument("--prob-map", required=True, help="Path to input probability GeoTIFF")
    parser.add_argument("--slope", default=None, help="Path to terrain slope GeoTIFF (degrees)")
    parser.add_argument("--pre-ndvi", default=None, help="Optional pre-event NDVI GeoTIFF")
    parser.add_argument("--pre-bsi", default=None, help="Optional pre-event BSI GeoTIFF")
    parser.add_argument("--threshold", type=float, default=0.50, help="Probability classification threshold")
    parser.add_argument("--min-slope", type=float, default=5.0, help="Minimum slope in degrees (default: 5°)")
    parser.add_argument("--max-slope", type=float, default=70.0, help="Maximum slope in degrees (default: 70°)")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum connected component size (pixels)")
    parser.add_argument("--output", default="output/landslide_response_map.tif", help="Output GeoTIFF path")
    args = parser.parse_args()

    print("=" * 70)
    print("Operational Landslide Post-Processing Pipeline")
    print("=" * 70)

    with rasterio.open(args.prob_map) as ds:
        prob = ds.read(1)
        profile = ds.profile

    slope_deg = None
    if args.slope:
        print(f"Loading slope raster: {args.slope}")
        slope_deg = resample_raster_to_reference(args.slope, args.prob_map)
    else:
        print("Auto-fetching elevation from AWS terrain tiles...")
        dem = fetch_dem_for_raster(args.prob_map)
        if dem is not None:
            slope_deg = compute_terrain_slope(dem)

    pre_ndvi = resample_raster_to_reference(args.pre_ndvi, args.prob_map) if args.pre_ndvi else None
    pre_bsi = resample_raster_to_reference(args.pre_bsi, args.prob_map) if args.pre_bsi else None

    print(f"Assembling multi-class landslide response map (threshold={args.threshold:.2f})...")
    resp_map, stats = generate_landslide_response_map(
        prob_map=prob,
        slope_deg=slope_deg,
        pre_ndvi=pre_ndvi,
        pre_bsi=pre_bsi,
        threshold=args.threshold,
        min_slope=args.min_slope,
        max_slope=args.max_slope,
        min_component_size=args.min_size,
    )

    save_landslide_map(resp_map, profile, args.output)

    print("\n--- Landslide Detection Summary ---")
    print(f"  Newly detected landslides:  {stats['new_landslide_pixels']:,} px ({stats['new_landslide_area_sqkm']:.3f} km²)")
    print(f"  Pre-existing bare rock:     {stats['preexisting_bare_pixels']:,} px ({stats['preexisting_bare_area_sqkm']:.3f} km²)")
    print(f"  Stable terrain:             {stats['stable_pixels']:,} px")
    print(f"  Excluded (slope <5°/>70°):  {stats['slope_excluded_pixels']:,} px")
    print(f"  Total valid terrain:        {stats['total_valid_pixels']:,} px")
    print("=" * 70)


if __name__ == "__main__":
    main()

