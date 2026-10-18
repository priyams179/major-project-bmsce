#!/usr/bin/env python3
"""
Operational Disaster Response Post-Processing Module

Provides:
  1. DEM Slope Masking: Eliminates mountain radar shadows and layover false positives (slope > 5.0 deg).
  2. Permanent Water Masking: Differentiates active flood inundation (Class 1) from permanent water bodies (Class 2).
  3. Multi-Class Disaster Map Generation (0: Dry, 1: Flood, 2: Permanent Water, 255: Steep Slope).
"""

import math
import os
import sys
import urllib.request
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy import ndimage


# ---------------------------------------------------------------------------
# 1. Terrain Slope Calculation (Horn's Algorithm)
# ---------------------------------------------------------------------------
def compute_terrain_slope(dem: np.ndarray,
                          pixel_size_x_m: float = 10.0,
                          pixel_size_y_m: float = 10.0) -> np.ndarray:
    """
    Compute terrain slope in degrees using Horn's algorithm (standard GDAL/ArcGIS formula).

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
    # 3x3 Sobel kernel for gradient estimation
    # dx: [-1 0 1; -2 0 2; -1 0 1] / (8 * dx)
    # dy: [-1 -2 -1; 0 0 0; 1 2 1] / (8 * dy)
    kernel_x = np.array([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]], dtype=np.float32) / (8.0 * max(pixel_size_x_m, 1e-3))

    kernel_y = np.array([[-1.0, -2.0, -1.0],
                         [ 0.0,  0.0,  0.0],
                         [ 1.0,  2.0,  1.0]], dtype=np.float32) / (8.0 * max(pixel_size_y_m, 1e-3))

    dem_clean = dem.astype(np.float32)
    unknown = np.isnan(dem_clean)
    if unknown.all():
        return np.zeros(dem_clean.shape, dtype=np.float32)
    if unknown.any():
        # unknown cells take the nearest known height - filling them with 0 would
        # put a cliff at the DEM's edge and mask it all as steep
        idx = ndimage.distance_transform_edt(unknown, return_distances=False, return_indices=True)
        dem_clean = dem_clean[tuple(idx)]

    dz_dx = ndimage.convolve(dem_clean, kernel_x, mode="reflect")
    dz_dy = ndimage.convolve(dem_clean, kernel_y, mode="reflect")

    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.arctan(rise_run) * (180.0 / np.pi)
    return slope_deg.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Resample / Match DEM or Water Mask to Target Raster
# ---------------------------------------------------------------------------
def resample_raster_to_reference(src_path: str, ref_path: str) -> np.ndarray:
    """
    Reproject and resample a source raster (DEM or Water mask) to match
    the CRS, dimensions, and transform of a reference raster.
    """
    with rasterio.open(ref_path) as ref_ds:
        return resample_raster_to_grid(src_path, ref_ds.transform, ref_ds.shape, ref_ds.crs)


def resample_raster_to_grid(src_path: str, dst_transform, shape, dst_crs, band: int = 1,
                            resampling=Resampling.bilinear) -> np.ndarray:
    """
    One band of a raster reprojected onto an arbitrary grid (transform + shape + CRS),
    NaN where the source has no data. Lets a large scene be processed block by block.
    """
    destination = np.full(shape, np.nan, dtype=np.float32)

    with rasterio.open(src_path) as src_ds:
        reproject(
            source=rasterio.band(src_ds, band),
            destination=destination,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
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
    with rasterio.open(ref_path) as ds:
        return fetch_dem_on_grid(ds.transform, ds.shape, ds.crs, cache_dir)


def fetch_dem_on_grid(dst_transform, shape, dst_crs, cache_dir: str = "data/dem_cache",
                      zoom: int = 10, max_tiles: int = 16) -> Optional[np.ndarray]:
    """
    Elevation (m) on an arbitrary grid from AWS terrain tiles, NaN where unknown.

    Every tile overlapping the grid is used, not just the one under its centre:
    a single zoom-10 tile is ~40 km wide, so a larger scene would otherwise have
    no DEM at its edges. The zoom drops until the grid needs at most max_tiles.
    Returns None if nothing could be fetched.
    """
    os.makedirs(cache_dir, exist_ok=True)
    try:
        h, w = shape
        left, top = dst_transform * (0, 0)
        right, bottom = dst_transform * (w, h)
        bounds = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
        if dst_crs and not dst_crs.is_geographic:
            bounds = transform_bounds(dst_crs, "EPSG:4326", *bounds)
        min_x, min_y, max_x, max_y = bounds

        while True:
            x0, y0 = deg2num(max_y, min_x, zoom)
            x1, y1 = deg2num(min_y, max_x, zoom)
            if (x1 - x0 + 1) * (y1 - y0 + 1) <= max_tiles or zoom <= 6:
                break
            zoom -= 1

        tiles = []
        for xtile in range(x0, x1 + 1):
            for ytile in range(y0, y1 + 1):
                tile_url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{xtile}/{ytile}.tif"
                cached_tile = os.path.join(cache_dir, f"dem_{zoom}_{xtile}_{ytile}.tif")
                if not os.path.exists(cached_tile) or os.path.getsize(cached_tile) == 0:
                    req = urllib.request.Request(tile_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp, open(cached_tile + ".tmp", "wb") as out:
                        out.write(resp.read())
                    os.replace(cached_tile + ".tmp", cached_tile)
                tiles.append(cached_tile)

        # mosaic first, then resample once - resampling tile by tile leaves a seam
        # of edge pixels along every tile boundary that reads as a cliff
        from rasterio.merge import merge
        srcs = [rasterio.open(p) for p in tiles]
        try:
            mosaic, mosaic_tr = merge(srcs)
            src_crs = srcs[0].crs
        finally:
            for s in srcs:
                s.close()
        mosaic = mosaic[0].astype(np.float32)
        mosaic[mosaic < -500] = np.nan                  # tile nodata

        destination = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=mosaic,
            destination=destination,
            src_transform=mosaic_tr,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs if dst_crs else "EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return None if np.isnan(destination).all() else destination
    except Exception as e:
        print(f"  [Warning] DEM fetch failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 4. Multi-Class Disaster Response Map Assembler
# ---------------------------------------------------------------------------
def generate_disaster_response_map(
    prob_map: np.ndarray,
    slope_deg: Optional[np.ndarray] = None,
    max_slope: float = 5.0,
    perm_water_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_component_size: int = 8,
    valid_mask: Optional[np.ndarray] = None,
    pixel_area_m2: float = 100.0,
) -> Tuple[np.ndarray, dict]:
    """
    Assemble the final multi-class disaster response map.

    Classes:
      0   : Dry Land
      1   : Active Inundated Flood Water
      2   : Permanent Water Body (Rivers, Lakes, Ocean)
      255 : Excluded Steep Terrain (Mountain Radar Shadow), or no radar data

    valid_mask marks pixels that have SAR data. Outside it the input was NaN /
    nodata, which normalises to -35 dB and looks exactly like water to the
    model, so those pixels are excluded rather than reported as flood.

    Returns
    -------
    disaster_map : np.ndarray (H, W), dtype uint8
    stats : dict
    """
    H, W = prob_map.shape
    raw_water = prob_map > threshold
    if valid_mask is not None:
        raw_water &= valid_mask

    # 1. Connected component filter (remove isolated noise specks)
    if min_component_size > 1:
        labeled, num = ndimage.label(raw_water, structure=np.ones((3, 3), dtype=bool))
        if num > 0:
            counts = np.bincount(labeled.ravel())
            keep = counts >= min_component_size
            keep[0] = False
            raw_water = keep[labeled]

    disaster_map = np.zeros((H, W), dtype=np.uint8)

    # 2. Permanent water separation
    if perm_water_mask is not None:
        if perm_water_mask.shape != (H, W):
            ph, pw = perm_water_mask.shape
            min_h, min_w = min(H, ph), min(W, pw)
            aligned_perm = np.zeros((H, W), dtype=bool)
            aligned_perm[:min_h, :min_w] = perm_water_mask[:min_h, :min_w] > 0.5
            perm_water_mask = aligned_perm

        is_perm = (perm_water_mask > 0.5) & raw_water
        is_flood = raw_water & (~is_perm)
        disaster_map[is_perm] = 2
        disaster_map[is_flood] = 1
    else:
        # If no permanent water mask provided, all detected water is flagged
        disaster_map[raw_water] = 1

    # 3. DEM Slope masking (> max_slope excluded)
    if slope_deg is not None:
        if slope_deg.shape != (H, W):
            sh, sw = slope_deg.shape
            min_h, min_w = min(H, sh), min(W, sw)
            aligned_slope = np.zeros((H, W), dtype=np.float32)
            aligned_slope[:min_h, :min_w] = slope_deg[:min_h, :min_w]
            slope_deg = aligned_slope

        is_steep = slope_deg > max_slope
        disaster_map[is_steep] = 255

    # 4. No SAR data -> unknown
    nodata_pixels = 0
    if valid_mask is not None:
        nodata_pixels = int((~valid_mask).sum())
        disaster_map[~valid_mask] = 255

    # Statistics
    total_valid = (disaster_map != 255).sum()
    flood_pixels = int((disaster_map == 1).sum())
    perm_pixels = int((disaster_map == 2).sum())
    dry_pixels = int((disaster_map == 0).sum())
    slope_masked_pixels = int((disaster_map == 255).sum()) - nodata_pixels

    stats = {
        "dry_pixels": dry_pixels,
        "flood_pixels": flood_pixels,
        "permanent_water_pixels": perm_pixels,
        "slope_masked_pixels": slope_masked_pixels,
        "nodata_pixels": nodata_pixels,
        "total_valid_pixels": int(total_valid),
        "flood_area_sqkm": (flood_pixels * pixel_area_m2) / 1e6,
        "permanent_water_area_sqkm": (perm_pixels * pixel_area_m2) / 1e6,
    }

    return disaster_map, stats

