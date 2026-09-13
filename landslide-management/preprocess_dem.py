#!/usr/bin/env python3
"""
Copernicus DEM GLO-30 Terrain Feature Extraction for Landslide Detection

Computes geomorphic conditioning factors critical for landslide susceptibility:
  - Slope (Horn's algorithm) — driving gravitational shear stress
  - Aspect (sin/cos encoded) — sunlight exposure and moisture retention
  - Profile Curvature — flow acceleration
  - Planform Curvature — flow convergence (pore-water accumulation)
  - TWI (Topographic Wetness Index) — hydrological accumulation
  - TRI (Terrain Ruggedness Index) — surface roughness

For India: >80% of landslides occur on slopes between 20-45 degrees.
Wayanad  (11.5N, 76.1E): DEM tile N11_E076.
Chamoli  (30.4N, 79.7E): DEM tile N30_E079.

Usage:
    python preprocess_dem.py --aoi wayanad --target-raster data/raw/sar/scene.tif --output output/dem/
    python preprocess_dem.py --bbox 75.9 11.3 76.3 11.8 --target-raster scene.tif --output output/dem/
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy import ndimage

# ---------------------------------------------------------------------------
# Pre-configured AOI bounding boxes for Indian landslide hotspots
# Format: (min_lon, min_lat, max_lon, max_lat)
# ---------------------------------------------------------------------------
AOI_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "wayanad": (75.9, 11.3, 76.3, 11.8),     # Kerala, Western Ghats
    "chamoli": (79.4, 30.2, 80.0, 30.6),     # Uttarakhand, Himalayas
    "shimla":  (76.7, 30.8, 77.3, 31.3),     # Himachal Pradesh
    "mangan":  (88.3, 27.3, 88.8, 27.8),     # Sikkim, Eastern Himalayas
    "noney":   (93.4, 24.5, 94.0, 25.0),     # Manipur, Northeast India
    "raigad":  (72.9, 18.7, 73.5, 19.2),     # Maharashtra, Western Ghats
}

# Copernicus DEM GLO-30 tile URL template (AWS Open Data)
COPERNICUS_DEM_URL_TEMPLATE = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
)
COPERNICUS_DEM_FILENAME_TEMPLATE = (
    "Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM.tif"
)

# AWS elevation tile URL (fallback, zoom-10 Web Mercator tiles)
AWS_ELEVATION_URL_TEMPLATE = (
    "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{x}/{y}.tif"
)

COMPRESS_OPTS: Dict = dict(
    compress="deflate",
    tiled=True,
    blockxsize=512,
    blockysize=512,
)


# ===========================================================================
# 1. Copernicus DEM tile fetching
# ===========================================================================

def _tile_filename(lat: int, lon: int) -> str:
    """Return the canonical Copernicus DEM GLO-30 tile filename."""
    return COPERNICUS_DEM_FILENAME_TEMPLATE.format(lat=lat, lon=lon)


def _tile_url(lat: int, lon: int) -> str:
    """Return the AWS S3 URL for a Copernicus DEM GLO-30 tile."""
    return COPERNICUS_DEM_URL_TEMPLATE.format(lat=lat, lon=lon)


def _download_file(url: str, dest_path: str, timeout: int = 60) -> None:
    """
    Download a file from *url* to *dest_path* using an atomic write.

    Uses a temporary ``.tmp`` file during download to avoid partial files.

    Parameters
    ----------
    url : str
        Remote URL.
    dest_path : str
        Local destination path.
    timeout : int
        HTTP request timeout in seconds.

    Raises
    ------
    urllib.error.URLError
        On connection errors.
    """
    tmp_path = dest_path + ".tmp"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "major-project-landslide/1.0 (python/urllib)"},
    )
    print(f"  [download] {os.path.basename(dest_path)} from {url}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _deg2tile(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """Convert (lat, lon) to Web Mercator tile (x, y) indices at *zoom*."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_copernicus_dem_tiles(
    bbox_wgs84: Tuple[float, float, float, float],
    cache_dir: str = "data/dem_cache",
) -> List[str]:
    """
    Download all Copernicus DEM GLO-30 tiles covering a WGS84 bounding box.

    Copernicus DEM GLO-30 tiles are 1°×1° in geographic coordinates.
    Tile naming convention: N{lat:02d}_E{lon:03d} (integer south-west corner).

    Parameters
    ----------
    bbox_wgs84 : tuple (min_lon, min_lat, max_lon, max_lat)
        Bounding box in WGS84 decimal degrees.
    cache_dir : str
        Directory to cache downloaded tiles.

    Returns
    -------
    tile_paths : list of str
        Absolute paths to locally cached tile GeoTIFFs.

    Notes
    -----
    Falls back to the AWS elevation tile service (zoom 10 Web Mercator) if
    a Copernicus tile download fails.  This mirrors the strategy used in
    ``postprocess.py::fetch_dem_for_raster``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    # Enumerate all 1°×1° tiles whose south-west corner falls within the bbox
    lat_range = range(int(math.floor(min_lat)), int(math.ceil(max_lat)))
    lon_range = range(int(math.floor(min_lon)), int(math.ceil(max_lon)))

    tile_paths: List[str] = []
    for lat in lat_range:
        for lon in lon_range:
            filename = _tile_filename(lat, lon)
            local_path = os.path.join(cache_dir, filename)

            if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
                print(f"  [cache hit] {filename}")
                tile_paths.append(local_path)
                continue

            # Try Copernicus DEM GLO-30 on AWS
            url = _tile_url(lat, lon)
            try:
                _download_file(url, local_path)
                tile_paths.append(local_path)
                continue
            except Exception as cop_err:
                print(
                    f"  [warning] Copernicus DEM download failed for "
                    f"N{lat:02d}_E{lon:03d}: {cop_err}"
                )

            # Fallback: AWS elevation tiles (zoom 10 Web Mercator)
            center_lat = lat + 0.5
            center_lon = lon + 0.5
            zoom = 10
            tx, ty = _deg2tile(center_lat, center_lon, zoom)
            fallback_url = AWS_ELEVATION_URL_TEMPLATE.format(
                zoom=zoom, x=tx, y=ty
            )
            fallback_path = os.path.join(
                cache_dir, f"aws_dem_{zoom}_{tx}_{ty}.tif"
            )
            try:
                if not (os.path.isfile(fallback_path) and
                        os.path.getsize(fallback_path) > 0):
                    _download_file(fallback_url, fallback_path)
                tile_paths.append(fallback_path)
            except Exception as aws_err:
                print(
                    f"  [error] Both Copernicus and AWS DEM downloads failed "
                    f"for tile N{lat:02d}_E{lon:03d}: {aws_err}"
                )

    if not tile_paths:
        raise RuntimeError(
            f"No DEM tiles could be fetched for bbox {bbox_wgs84}. "
            "Check your network connection and bounding box coordinates."
        )

    print(f"  [fetch] {len(tile_paths)} DEM tile(s) available.")
    return tile_paths


# ===========================================================================
# 2. Tile merging and reprojection
# ===========================================================================

def merge_and_reproject_dem(
    tile_paths: List[str],
    target_crs,
    target_transform,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Merge multiple DEM tiles and reproject to a target CRS and grid.

    Parameters
    ----------
    tile_paths : list of str
        Paths to input DEM GeoTIFFs (may have different extents / CRS).
    target_crs : rasterio.crs.CRS or str
        Target coordinate reference system.
    target_transform : rasterio.transform.Affine
        Affine transform of the target grid.
    target_shape : (H, W)
        Height and width of the target grid in pixels.

    Returns
    -------
    dem : np.ndarray, shape (H, W), dtype float32
        Elevation in metres, aligned to the target grid.
        NaN where no data is available.
    """
    H, W = target_shape

    # Open all tiles for merging
    open_files = [rasterio.open(p) for p in tile_paths]
    try:
        merged_data, merged_transform = rasterio_merge(open_files)
        merged_src_crs = open_files[0].crs
    finally:
        for f in open_files:
            f.close()

    # merged_data is (bands, H, W); we only need band 1
    merged_2d = merged_data[0].astype(np.float32)

    # Replace nodata (often -9999 or 0) with NaN
    nodata_candidates = [-9999.0, -32768.0, 0.0]
    with rasterio.open(tile_paths[0]) as src:
        src_nodata = src.nodata
    if src_nodata is not None:
        merged_2d[merged_2d == src_nodata] = np.nan

    destination = np.full((H, W), np.nan, dtype=np.float32)
    reproject(
        source=merged_2d[np.newaxis, :, :],  # (1, H_src, W_src)
        destination=destination[np.newaxis, :, :],
        src_transform=merged_transform,
        src_crs=merged_src_crs if merged_src_crs else "EPSG:4326",
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )

    return destination


# ===========================================================================
# 3. Slope — Horn's algorithm (exact port from postprocess.py)
# ===========================================================================

def compute_slope(
    dem: np.ndarray,
    pixel_size_x_m: float = 30.0,
    pixel_size_y_m: float = 30.0,
) -> np.ndarray:
    """
    Compute terrain slope in degrees using Horn's algorithm.

    This is the standard GDAL / ArcGIS formula, adapted from
    ``postprocess.py::compute_terrain_slope`` to use the 30 m GLO-30 pixel
    spacing by default.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation array in metres.  NaN values are filled with local mean.
    pixel_size_x_m, pixel_size_y_m : float
        Pixel spacing in metres (east–west and north–south respectively).

    Returns
    -------
    slope_deg : np.ndarray (H, W), dtype float32
        Slope in degrees [0, 90].
    """
    # Sobel-based Horn gradient kernels
    # dz/dx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / (8 * dx)
    # dz/dy: [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]] / (8 * dy)
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    ) / (8.0 * max(pixel_size_x_m, 1e-3))

    kernel_y = np.array(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=np.float64,
    ) / (8.0 * max(pixel_size_y_m, 1e-3))

    dem_clean = np.nan_to_num(dem.astype(np.float64), nan=0.0)

    dz_dx = ndimage.convolve(dem_clean, kernel_x, mode="reflect")
    dz_dy = ndimage.convolve(dem_clean, kernel_y, mode="reflect")

    rise_run = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_deg = np.degrees(np.arctan(rise_run))
    return slope_deg.astype(np.float32)


# ===========================================================================
# 4. Aspect (sin/cos encoded to avoid 0/360° discontinuity)
# ===========================================================================

def compute_aspect(
    dem: np.ndarray,
    pixel_size_x_m: float = 30.0,
    pixel_size_y_m: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute terrain aspect and return it as circular (sin, cos) components.

    Circular encoding avoids the 0°/360° discontinuity that would cause
    north-facing slopes to appear as two separate classes in a feature space.
    Aspect = 0° points North; increases clockwise.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation array in metres.
    pixel_size_x_m, pixel_size_y_m : float
        Pixel spacing in metres.

    Returns
    -------
    sin_aspect : np.ndarray (H, W), dtype float32, values in [−1, 1]
    cos_aspect : np.ndarray (H, W), dtype float32, values in [−1, 1]
    """
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    ) / (8.0 * max(pixel_size_x_m, 1e-3))

    kernel_y = np.array(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        dtype=np.float64,
    ) / (8.0 * max(pixel_size_y_m, 1e-3))

    dem_clean = np.nan_to_num(dem.astype(np.float64), nan=0.0)

    dz_dx = ndimage.convolve(dem_clean, kernel_x, mode="reflect")
    dz_dy = ndimage.convolve(dem_clean, kernel_y, mode="reflect")

    # Aspect in radians: atan2(−dz/dx, dz/dy) measured clockwise from North
    aspect_rad = np.arctan2(-dz_dx, dz_dy)

    sin_aspect = np.sin(aspect_rad).astype(np.float32)
    cos_aspect = np.cos(aspect_rad).astype(np.float32)
    return sin_aspect, cos_aspect


# ===========================================================================
# 5. Profile curvature
# ===========================================================================

def compute_profile_curvature(
    dem: np.ndarray,
    pixel_size_m: float = 30.0,
) -> np.ndarray:
    """
    Compute profile curvature (rate of change of slope along flow direction).

    Profile curvature is positive on convex slopes (flow acceleration) and
    negative on concave slopes (deposition zones).  It is computed using
    second-order finite differences of the elevation field.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation in metres.
    pixel_size_m : float
        Isotropic pixel spacing in metres.

    Returns
    -------
    profile_curv : np.ndarray (H, W), dtype float32
    """
    z = np.nan_to_num(dem.astype(np.float64), nan=0.0)
    h = pixel_size_m

    # First-order gradients (central differences)
    dz_dx = ndimage.convolve(
        z,
        np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64) / (2.0 * h),
        mode="reflect",
    )
    dz_dy = ndimage.convolve(
        z,
        np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64) / (2.0 * h),
        mode="reflect",
    )

    # Second-order gradients
    d2z_dx2 = ndimage.convolve(
        z,
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float64) / (h * h),
        mode="reflect",
    )
    d2z_dy2 = ndimage.convolve(
        z,
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float64) / (h * h),
        mode="reflect",
    )
    d2z_dxdy = ndimage.convolve(
        z,
        np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64) / (4.0 * h * h),
        mode="reflect",
    )

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = d2z_dxdy
    t = d2z_dy2

    # Profile curvature formula (Zevenbergen & Thorne, 1987)
    denom = (p ** 2 + q ** 2) * np.sqrt(1.0 + p ** 2 + q ** 2)
    numer = (
        p ** 2 * r
        + 2.0 * p * q * s
        + q ** 2 * t
    )
    # Avoid division by zero on flat terrain
    profile_curv = np.where(np.abs(denom) > 1e-10, numer / denom, 0.0)

    return profile_curv.astype(np.float32)


# ===========================================================================
# 6. Planform curvature
# ===========================================================================

def compute_planform_curvature(
    dem: np.ndarray,
    pixel_size_m: float = 30.0,
) -> np.ndarray:
    """
    Compute planform curvature (curvature perpendicular to the gradient).

    Planform curvature is positive on convex terrain (divergent flow) and
    negative on concave terrain (convergent flow / pore-water accumulation).
    Negative planform curvature is a strong landslide conditioning factor.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation in metres.
    pixel_size_m : float
        Isotropic pixel spacing in metres.

    Returns
    -------
    planform_curv : np.ndarray (H, W), dtype float32
    """
    z = np.nan_to_num(dem.astype(np.float64), nan=0.0)
    h = pixel_size_m

    dz_dx = ndimage.convolve(
        z,
        np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64) / (2.0 * h),
        mode="reflect",
    )
    dz_dy = ndimage.convolve(
        z,
        np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64) / (2.0 * h),
        mode="reflect",
    )
    d2z_dx2 = ndimage.convolve(
        z,
        np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float64) / (h * h),
        mode="reflect",
    )
    d2z_dy2 = ndimage.convolve(
        z,
        np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float64) / (h * h),
        mode="reflect",
    )
    d2z_dxdy = ndimage.convolve(
        z,
        np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64) / (4.0 * h * h),
        mode="reflect",
    )

    p = dz_dx
    q = dz_dy
    r = d2z_dx2
    s = d2z_dxdy
    t = d2z_dy2

    # Planform curvature formula (Zevenbergen & Thorne, 1987)
    numer = (
        p ** 2 * t
        - 2.0 * p * q * s
        + q ** 2 * r
    )
    denom = (p ** 2 + q ** 2) * np.sqrt(1.0 + p ** 2 + q ** 2)

    planform_curv = np.where(np.abs(denom) > 1e-10, numer / denom, 0.0)

    return planform_curv.astype(np.float32)


# ===========================================================================
# 7. TWI — Topographic Wetness Index
# ===========================================================================

def compute_twi(
    slope_rad: np.ndarray,
    flow_accumulation: Optional[np.ndarray] = None,
    pixel_size_m: float = 30.0,
) -> np.ndarray:
    """
    Compute the Topographic Wetness Index (TWI).

    TWI = ln(α / tan(β))

    where α is the upslope contributing area (m²) and β is slope in radians.
    Larger TWI → greater soil moisture → higher pore-water pressure → more
    susceptible to failure.

    Parameters
    ----------
    slope_rad : np.ndarray (H, W)
        Slope in radians (output of ``np.radians(compute_slope(dem))``).
    flow_accumulation : np.ndarray (H, W) or None
        Upslope contributing area in pixels.  If None, a simple approximation
        is used: α = (1 + |flow_accum|) * pixel_size_m² where flow_accum is
        approximated as a spatially uniform unit catchment.
    pixel_size_m : float
        Pixel size in metres (used to convert pixel count → area in m²).

    Returns
    -------
    twi : np.ndarray (H, W), dtype float32
        TWI values, clipped to [0, 25] to remove extreme outliers.
    """
    pixel_area_m2 = pixel_size_m * pixel_size_m

    if flow_accumulation is not None:
        # Contributing area in m²
        alpha = (np.maximum(flow_accumulation, 0.0) + 1.0) * pixel_area_m2
    else:
        # Approximation: constant unit catchment for each pixel (1 cell)
        # In the absence of a proper D8/D-infinity flow routing,
        # we use a spatially uniform upslope area approximation.
        alpha = np.ones_like(slope_rad, dtype=np.float64) * pixel_area_m2

    # Avoid tan(0) → division by zero on flat terrain
    slope_clamped = np.maximum(np.abs(slope_rad), 1e-4)  # min ~0.006°
    tan_beta = np.tan(slope_clamped)

    twi = np.log(alpha / tan_beta)
    # Clip extreme values (very flat, very large catchment areas)
    twi = np.clip(twi, 0.0, 25.0)

    return twi.astype(np.float32)


# ===========================================================================
# 8. TRI — Terrain Ruggedness Index
# ===========================================================================

def compute_tri(dem: np.ndarray) -> np.ndarray:
    """
    Compute the Terrain Ruggedness Index (TRI).

    TRI = mean absolute difference between a cell's elevation and its 8
    immediate neighbours (Riley et al., 1999).  High TRI indicates rough,
    broken terrain typical of active landslide zones.

    Parameters
    ----------
    dem : np.ndarray (H, W)
        Elevation in metres.

    Returns
    -------
    tri : np.ndarray (H, W), dtype float32
    """
    z = np.nan_to_num(dem.astype(np.float64), nan=0.0)

    # 8-neighbour offsets
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    acc = np.zeros_like(z)
    for dy, dx in offsets:
        # Shift the DEM by (dy, dx) with reflect padding
        padded = np.pad(z, 1, mode="reflect")
        shifted = padded[1 + dy : 1 + dy + z.shape[0],
                         1 + dx : 1 + dx + z.shape[1]]
        acc += np.abs(z - shifted)

    tri = acc / 8.0
    return tri.astype(np.float32)


# ===========================================================================
# 9. Feature normalization
# ===========================================================================

def normalize_feature(
    arr: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize a 2-D feature array to [0, 1].

    Parameters
    ----------
    arr : np.ndarray
        Input array (any shape).
    method : str
        ``"minmax"`` — global min-max normalization.
        ``"percentile"`` — clip to [p2, p98] then min-max normalize.
            Robust against outliers from water bodies / cliffs.

    Returns
    -------
    normalized : np.ndarray, dtype float32, values in [0, 1]
    """
    a = arr.astype(np.float64)
    valid = a[np.isfinite(a)]

    if valid.size == 0:
        return np.zeros_like(a, dtype=np.float32)

    if method == "percentile":
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
    elif method == "minmax":
        lo = float(valid.min())
        hi = float(valid.max())
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    if abs(hi - lo) < 1e-10:
        return np.zeros_like(a, dtype=np.float32)

    normalized = (np.clip(a, lo, hi) - lo) / (hi - lo)
    return normalized.astype(np.float32)


# ===========================================================================
# 10. Master terrain feature extractor
# ===========================================================================

def extract_all_terrain_features(
    dem_path_or_array,
    output_dir: str,
    pixel_size_m: float = 30.0,
    profile: Optional[dict] = None,
) -> Dict[str, str]:
    """
    Compute all terrain conditioning factors and save as GeoTIFFs.

    Parameters
    ----------
    dem_path_or_array : str or np.ndarray
        Either a path to a DEM GeoTIFF or an already-loaded (H, W) array.
        If a path is given, the spatial reference is read from it.
    output_dir : str
        Directory to write output GeoTIFFs.
    pixel_size_m : float
        Pixel size in metres (used only when ``dem_path_or_array`` is an array).
    profile : dict or None
        Rasterio profile to use when writing GeoTIFFs.  Required when
        ``dem_path_or_array`` is a NumPy array.

    Returns
    -------
    paths : dict
        Maps feature name → absolute path to the output GeoTIFF.
        Keys: ``elevation``, ``slope``, ``sin_aspect``, ``cos_aspect``,
              ``profile_curv``, ``planform_curv``, ``twi``, ``tri``.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ----- Load DEM --------------------------------------------------------
    if isinstance(dem_path_or_array, (str, os.PathLike)):
        dem_path = str(dem_path_or_array)
        with rasterio.open(dem_path) as ds:
            dem = ds.read(1).astype(np.float32)
            profile = ds.profile.copy()
            # Derive pixel spacing from transform (in metres if CRS is projected)
            res = ds.res  # (height, width) in CRS units
            if ds.crs and not ds.crs.is_geographic:
                pixel_size_m = float(np.mean(res))
            else:
                # Geographic CRS: convert ~arcsec to metres (approx 30m for GLO-30)
                pixel_size_m = pixel_size_m  # caller-specified
    else:
        dem = np.asarray(dem_path_or_array, dtype=np.float32)
        if profile is None:
            raise ValueError(
                "A rasterio profile must be provided when dem_path_or_array "
                "is a NumPy array."
            )

    H, W = dem.shape
    print(f"\n[terrain features] DEM shape: {W} × {H}, pixel_size={pixel_size_m:.1f}m")

    def _save(name: str, data: np.ndarray) -> str:
        out_path = os.path.join(output_dir, f"{name}.tif")
        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype="float32",
            nodata=float("nan"),
            **{k: v for k, v in COMPRESS_OPTS.items()},
        )
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(data.astype(np.float32), 1)
        print(f"  Saved: {out_path}")
        return out_path

    paths: Dict[str, str] = {}

    # -- Elevation --------------------------------------------------------
    print("  Computing elevation …")
    elev_norm = normalize_feature(dem, method="percentile")
    paths["elevation"] = _save("elevation", elev_norm)

    # -- Slope ------------------------------------------------------------
    print("  Computing slope (Horn's algorithm) …")
    slope_deg = compute_slope(dem, pixel_size_x_m=pixel_size_m, pixel_size_y_m=pixel_size_m)
    slope_norm = normalize_feature(slope_deg, method="percentile")
    paths["slope"] = _save("slope", slope_norm)

    # -- Aspect -----------------------------------------------------------
    print("  Computing aspect (sin/cos encoding) …")
    sin_asp, cos_asp = compute_aspect(
        dem, pixel_size_x_m=pixel_size_m, pixel_size_y_m=pixel_size_m
    )
    # sin/cos are already in [−1, 1]; normalize to [0, 1]
    paths["sin_aspect"] = _save("sin_aspect", (sin_asp + 1.0) / 2.0)
    paths["cos_aspect"] = _save("cos_aspect", (cos_asp + 1.0) / 2.0)

    # -- Profile curvature ------------------------------------------------
    print("  Computing profile curvature …")
    prof_curv = compute_profile_curvature(dem, pixel_size_m=pixel_size_m)
    prof_curv_norm = normalize_feature(prof_curv, method="percentile")
    paths["profile_curv"] = _save("profile_curv", prof_curv_norm)

    # -- Planform curvature -----------------------------------------------
    print("  Computing planform curvature …")
    plan_curv = compute_planform_curvature(dem, pixel_size_m=pixel_size_m)
    plan_curv_norm = normalize_feature(plan_curv, method="percentile")
    paths["planform_curv"] = _save("planform_curv", plan_curv_norm)

    # -- TWI --------------------------------------------------------------
    print("  Computing TWI …")
    slope_rad = np.radians(slope_deg)
    twi = compute_twi(slope_rad, flow_accumulation=None, pixel_size_m=pixel_size_m)
    twi_norm = normalize_feature(twi, method="percentile")
    paths["twi"] = _save("twi", twi_norm)

    # -- TRI --------------------------------------------------------------
    print("  Computing TRI …")
    tri = compute_tri(dem)
    tri_norm = normalize_feature(tri, method="percentile")
    paths["tri"] = _save("tri", tri_norm)

    print(f"  [terrain features] All {len(paths)} features saved.")
    return paths


# ===========================================================================
# 11. Full AOI pipeline
# ===========================================================================

def fetch_and_extract_for_aoi(
    bbox_wgs84: Tuple[float, float, float, float],
    target_raster_path: str,
    output_dir: str,
    cache_dir: str = "data/dem_cache",
    pixel_size_m: float = 30.0,
) -> Dict[str, str]:
    """
    Full pipeline: fetch DEM tiles → merge → reproject → extract features.

    Reprojects and aligns the DEM to exactly match the target raster so that
    terrain features can be used directly as input channels to the CNN.

    Parameters
    ----------
    bbox_wgs84 : (min_lon, min_lat, max_lon, max_lat)
        Bounding box for tile fetching in WGS84 decimal degrees.
    target_raster_path : str
        Path to a reference raster (SAR or optical) whose CRS, extent, and
        resolution defines the target grid.
    output_dir : str
        Directory to write DEM and terrain feature GeoTIFFs.
    cache_dir : str
        Directory for cached DEM tiles.
    pixel_size_m : float
        Nominal pixel size in metres (informational; actual pixel size is read
        from the target raster).

    Returns
    -------
    feature_paths : dict
        Same structure as ``extract_all_terrain_features``.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"[fetch_and_extract_for_aoi] bbox={bbox_wgs84}")
    print(f"  Target raster: {target_raster_path}")

    # -- Read target raster spatial reference -----------------------------
    with rasterio.open(target_raster_path) as ref:
        target_crs = ref.crs
        target_transform = ref.transform
        target_h = ref.height
        target_w = ref.width
        target_res = ref.res
        ref_profile = ref.profile.copy()
        ref_bounds = ref.bounds

    print(f"  Target CRS: {target_crs}")
    print(f"  Target size: {target_w} × {target_h}, res={target_res}")

    # If the target has no CRS, assume WGS84
    if target_crs is None:
        target_crs = "EPSG:4326"
        print("  [warning] Target raster has no CRS; assuming EPSG:4326.")

    # -- Expand bbox if needed to cover the target raster footprint ------
    # Convert target bounds to WGS84 to ensure DEM tiles are fetched
    try:
        if hasattr(target_crs, "is_geographic") and not target_crs.is_geographic:
            wgs_bounds = transform_bounds(target_crs, "EPSG:4326", *ref_bounds)
        else:
            wgs_bounds = ref_bounds
        # Union with user-supplied bbox
        expanded_bbox = (
            min(bbox_wgs84[0], wgs_bounds.left),
            min(bbox_wgs84[1], wgs_bounds.bottom),
            max(bbox_wgs84[2], wgs_bounds.right),
            max(bbox_wgs84[3], wgs_bounds.top),
        )
    except Exception:
        expanded_bbox = bbox_wgs84

    # -- Fetch DEM tiles --------------------------------------------------
    tile_paths = fetch_copernicus_dem_tiles(expanded_bbox, cache_dir=cache_dir)

    # -- Merge and reproject to target grid -------------------------------
    print("  Merging and reprojecting DEM tiles …")
    dem = merge_and_reproject_dem(
        tile_paths,
        target_crs=target_crs,
        target_transform=target_transform,
        target_shape=(target_h, target_w),
    )
    print(f"  DEM range: [{np.nanmin(dem):.1f}, {np.nanmax(dem):.1f}] m")

    # -- Save merged DEM --------------------------------------------------
    dem_out_path = os.path.join(output_dir, "dem_merged.tif")
    dem_profile = ref_profile.copy()
    dem_profile.update(
        count=1,
        dtype="float32",
        nodata=float("nan"),
        **{k: v for k, v in COMPRESS_OPTS.items()},
    )
    with rasterio.open(dem_out_path, "w", **dem_profile) as dst:
        dst.write(dem, 1)
    print(f"  Saved merged DEM: {dem_out_path}")

    # -- Extract terrain features ----------------------------------------
    # Derive pixel size from transform (works for projected CRS)
    px_size = float(np.mean(target_res))
    if target_crs and hasattr(target_crs, "is_geographic") and target_crs.is_geographic:
        # Approximate conversion from degrees to metres at the AOI latitude
        centre_lat = (bbox_wgs84[1] + bbox_wgs84[3]) / 2.0
        px_size = float(np.mean(target_res)) * math.pi / 180.0 * 6_371_000.0 * math.cos(
            math.radians(centre_lat)
        )
        px_size = max(px_size, 1.0)

    feature_paths = extract_all_terrain_features(
        dem,
        output_dir=output_dir,
        pixel_size_m=px_size,
        profile=dem_profile,
    )

    print(f"  [pipeline] Complete in {time.time() - t0:.1f}s")
    return feature_paths


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copernicus DEM GLO-30 Terrain Feature Extraction for Landslide Detection\n\n"
            "Downloads DEM tiles, merges and reprojects them to match a target raster,\n"
            "then computes slope, aspect, curvature, TWI, TRI and saves them as GeoTIFFs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    aoi_group = parser.add_mutually_exclusive_group(required=True)
    aoi_group.add_argument(
        "--aoi",
        choices=list(AOI_PROFILES.keys()),
        metavar="AOI_NAME",
        help=(
            "Named AOI shortcut.  One of: "
            + ", ".join(AOI_PROFILES.keys())
            + "."
        ),
    )
    aoi_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Custom bounding box in WGS84 decimal degrees.",
    )

    parser.add_argument(
        "--target-raster",
        required=True,
        metavar="RASTER_PATH",
        help=(
            "Reference raster (SAR or optical scene) whose CRS, extent and "
            "resolution define the output grid."
        ),
    )
    parser.add_argument(
        "--output",
        default="output/dem",
        metavar="OUTPUT_DIR",
        help="Output directory for terrain feature GeoTIFFs (default: output/dem).",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/dem_cache",
        metavar="CACHE_DIR",
        help="Directory for cached DEM tiles (default: data/dem_cache).",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=30.0,
        metavar="METRES",
        help=(
            "Nominal pixel size in metres used for gradient kernels when the "
            "CRS is geographic (default: 30.0 for GLO-30)."
        ),
    )

    args = parser.parse_args()

    if args.aoi:
        bbox = AOI_PROFILES[args.aoi]
        print(f"Using pre-configured AOI: {args.aoi} {bbox}")
    else:
        bbox = tuple(args.bbox)
        print(f"Using custom bbox: {bbox}")

    if not os.path.isfile(args.target_raster):
        parser.error(f"Target raster not found: {args.target_raster}")

    feature_paths = fetch_and_extract_for_aoi(
        bbox_wgs84=bbox,
        target_raster_path=args.target_raster,
        output_dir=args.output,
        cache_dir=args.cache_dir,
        pixel_size_m=args.pixel_size,
    )

    print(f"\n{'='*60}")
    print("Output terrain feature GeoTIFFs:")
    for name, path in feature_paths.items():
        print(f"  {name:20s} → {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
