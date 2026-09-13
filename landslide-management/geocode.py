#!/usr/bin/env python3
"""
Geocode Landslide Detection Maps to Geographic WGS84 (EPSG:4326)

Supports two complementary geocoding modes:
  1. Sentinel-1 Native Radar Coordinate Geocoding:
     Uses the geolocation grid tie-points from the SAFE annotation XML to build
     an inverse coordinate mapping and resample radar images onto a regular lat/lon grid.
     (No external SNAP or GDAL command-line binaries required).
  2. Projected CRS (UTM) to EPSG:4326 Reprojection:
     Direct high-performance bilinear/nearest-neighbor reprojection for multi-modal
     and optical landslide rasters.

Usage:
    python geocode.py --input output/landslide_response_map.tif --safe-dir data/S1A_IW_GRDH...SAFE
    python geocode.py --input output/landslide_response_map.tif --reproject
"""

import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

# Earth constants
M_PER_DEG_LAT = 111320.0  # Metres per degree latitude


# ---------------------------------------------------------------------------
# 1. Native Sentinel-1 Radar Geolocation Grid Parser
# ---------------------------------------------------------------------------
def parse_geolocation_grid(xml_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the geolocation grid from a Sentinel-1 annotation XML file.

    Returns
    -------
    unique_lines : np.ndarray (1D)
    unique_pixels : np.ndarray (1D)
    lats_grid : np.ndarray (2D)
    lons_grid : np.ndarray (2D)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gcp_list = root.find(".//geolocationGridPointList")
    if gcp_list is None:
        raise ValueError("No geolocationGridPointList found in annotation XML")

    raw_lines, raw_pixels, raw_lats, raw_lons = [], [], [], []
    for pt in gcp_list:
        raw_lines.append(int(pt.find("line").text))
        raw_pixels.append(int(pt.find("pixel").text))
        raw_lats.append(float(pt.find("latitude").text))
        raw_lons.append(float(pt.find("longitude").text))

    raw_lines = np.array(raw_lines)
    raw_pixels = np.array(raw_pixels)
    raw_lats = np.array(raw_lats)
    raw_lons = np.array(raw_lons)

    unique_lines = np.unique(raw_lines)
    unique_pixels = np.unique(raw_pixels)
    n_lines = len(unique_lines)
    n_pixels = len(unique_pixels)

    lats_grid = np.empty((n_lines, n_pixels), dtype=np.float64)
    lons_grid = np.empty((n_lines, n_pixels), dtype=np.float64)

    for i, ln in enumerate(unique_lines):
        for j, px in enumerate(unique_pixels):
            idx = np.where((raw_lines == ln) & (raw_pixels == px))[0][0]
            lats_grid[i, j] = raw_lats[idx]
            lons_grid[i, j] = raw_lons[idx]

    return unique_lines.astype(np.float64), unique_pixels.astype(np.float64), lats_grid, lons_grid


def build_inverse_interpolator(
    lines: np.ndarray,
    pixels: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[LinearNDInterpolator, LinearNDInterpolator]:
    """
    Build 2D interpolators mapping (lat, lon) -> (line, pixel).
    """
    points = np.column_stack([lats.ravel(), lons.ravel()])
    line_vals = np.repeat(lines, len(pixels))
    pixel_vals = np.tile(pixels, len(lines))

    interp_line = LinearNDInterpolator(points, line_vals, fill_value=np.nan)
    interp_pixel = LinearNDInterpolator(points, pixel_vals, fill_value=np.nan)
    return interp_line, interp_pixel


def compute_output_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    pixel_size_m: float = 10.0,
) -> Tuple[int, int, float, float, float, float, float, float]:
    """
    Determine regular output lat/lon grid bounds and resolution.
    """
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    lat_min -= 0.001
    lat_max += 0.001
    lon_min -= 0.001
    lon_max += 0.001

    center_lat = (lat_min + lat_max) / 2.0
    res_lat = pixel_size_m / M_PER_DEG_LAT
    res_lon = pixel_size_m / (M_PER_DEG_LAT * np.cos(np.radians(center_lat)))

    out_h = int(np.ceil((lat_max - lat_min) / res_lat))
    out_w = int(np.ceil((lon_max - lon_min) / res_lon))

    return out_h, out_w, res_lat, res_lon, lat_min, lat_max, lon_min, lon_max


def geocode_image(
    img: np.ndarray,
    interp_line: LinearNDInterpolator,
    interp_pixel: LinearNDInterpolator,
    out_h: int,
    out_w: int,
    res_lat: float,
    res_lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    chunk_rows: int = 500,
    order: int = 1,
) -> np.ndarray:
    """
    Resample image from radar geometry to regular geographic grid using chunked coordinates.
    """
    H, W = img.shape
    fill_val = 255 if np.issubdtype(img.dtype, np.integer) else np.nan
    geo_img = np.full((out_h, out_w), fill_val, dtype=np.float32)

    for row_start in range(0, out_h, chunk_rows):
        row_end = min(row_start + chunk_rows, out_h)
        out_lat = lat_max - (np.arange(row_start, row_end) + 0.5) * res_lat
        out_lon = lon_min + (np.arange(out_w) + 0.5) * res_lon
        out_lat_grid, out_lon_grid = np.meshgrid(out_lat, out_lon, indexing="ij")

        query = np.column_stack([out_lat_grid.ravel(), out_lon_grid.ravel()])
        in_lines = interp_line(query)
        in_pixels = interp_pixel(query)

        in_lines = in_lines.reshape(row_end - row_start, out_w)
        in_pixels = in_pixels.reshape(row_end - row_start, out_w)

        valid = (
            np.isfinite(in_lines)
            & np.isfinite(in_pixels)
            & (in_lines >= 0)
            & (in_lines < H - 1)
            & (in_pixels >= 0)
            & (in_pixels < W - 1)
        )

        if not valid.any():
            continue

        valid_rows, valid_cols = np.where(valid)
        coords = np.array([in_lines[valid_rows, valid_cols], in_pixels[valid_rows, valid_cols]])
        sampled = map_coordinates(
            img.astype(np.float64), coords, order=order, mode="constant", cval=fill_val
        )
        geo_img[row_start:row_end][valid_rows, valid_cols] = sampled.astype(np.float32)

    return geo_img


def save_geocoded_geotiff(
    path: str,
    data: np.ndarray,
    res_lat: float,
    res_lon: float,
    lat_min: float,
    lon_min: float,
    is_integer: bool = False,
) -> None:
    """Save geocoded array as EPSG:4326 GeoTIFF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = data.shape
    transform = from_origin(lon_min, lat_min + h * res_lat, res_lon, res_lat)
    gdtype = rasterio.uint8 if is_integer else rasterio.float32
    nodata = 255 if is_integer else float("nan")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=gdtype,
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.uint8 if is_integer else np.float32), 1)


def find_annotation_xml(safe_dir: str) -> str:
    """Find the annotation XML in a SAFE directory."""
    matches = (
        glob.glob(os.path.join(safe_dir, "annotation", "*vh*.xml"))
        + glob.glob(os.path.join(safe_dir, "annotation", "*vv*.xml"))
        + glob.glob(os.path.join(safe_dir, "annotation", "*.xml"))
    )
    ann_matches = [m for m in matches if os.path.dirname(m) == os.path.join(safe_dir, "annotation")]
    if not ann_matches:
        raise FileNotFoundError(f"No annotation XML found in {safe_dir}/annotation/")
    return ann_matches[0]


# ---------------------------------------------------------------------------
# 2. Standard Projected CRS (UTM) to EPSG:4326 Reprojection
# ---------------------------------------------------------------------------
def reproject_to_wgs84(src_path: str, dst_path: str) -> None:
    """
    Reproject any GeoTIFF with an existing CRS to geographic WGS84 (EPSG:4326).
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    dst_crs = "EPSG:4326"

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        is_int = np.issubdtype(src.dtypes[0], np.integer)
        resampling = Resampling.nearest if is_int else Resampling.bilinear

        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "deflate",
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )

    print(f"  Reprojected {src_path} -> {dst_path} (EPSG:4326)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Geocode Landslide Maps to EPSG:4326")
    parser.add_argument("--input", required=True, help="Input GeoTIFF path to geocode")
    parser.add_argument("--output", default=None, help="Output GeoTIFF path (default: <input>_wgs84.tif)")
    parser.add_argument("--safe-dir", default=None, help="Sentinel-1 SAFE directory for radar tie-point geocoding")
    parser.add_argument("--reproject", action="store_true", help="Directly reproject existing projected CRS to WGS84")
    parser.add_argument("--pixel-size", type=float, default=10.0, help="Output pixel size in metres (default: 10m)")
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        base, ext = os.path.splitext(args.input)
        out_path = f"{base}_wgs84{ext}"

    if args.reproject or args.safe_dir is None:
        print(f"Reprojecting {args.input} to EPSG:4326...")
        reproject_to_wgs84(args.input, out_path)
        print("Done!")
        return

    # Native radar geometry geocoding
    print(f"Geocoding radar image using SAFE geolocation grid: {args.safe_dir}")
    ann_xml = find_annotation_xml(args.safe_dir)
    lines, pixels, lats, lons = parse_geolocation_grid(ann_xml)
    interp_line, interp_pixel = build_inverse_interpolator(lines, pixels, lats, lons)
    out_h, out_w, res_lat, res_lon, lat_min, lat_max, lon_min, lon_max = compute_output_grid(
        lats, lons, pixel_size_m=args.pixel_size
    )

    with rasterio.open(args.input) as ds:
        img = ds.read(1)

    is_int = np.issubdtype(img.dtype, np.integer)
    order = 0 if is_int else 1

    t0 = time.time()
    geo_img = geocode_image(
        img, interp_line, interp_pixel,
        out_h, out_w, res_lat, res_lon,
        lat_min, lat_max, lon_min, lon_max,
        order=order,
    )
    print(f"Geocoded in {time.time() - t0:.1f}s")

    save_geocoded_geotiff(out_path, geo_img, res_lat, res_lon, lat_min, lon_min, is_integer=is_int)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

