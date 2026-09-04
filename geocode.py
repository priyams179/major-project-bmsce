#!/usr/bin/env python3
"""
Geocode Sentinel-1 GRD images from image coordinates to lat/lon (EPSG:4326).

Uses the geolocation grid from the SAFE annotation XML to build an inverse
coordinate mapping, then resamples the image onto a regular lat/lon grid.

No external software (SNAP, GDAL command-line) required.

Usage:
    .venv/bin/python geocode.py [--input-dir output] [--safe-dir data/...]
    .venv/bin/python geocode.py --input output/flood_map.tif
"""

import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates

# Earth constants
M_PER_DEG_LAT = 111320.0  # metres per degree latitude (approximate)


def parse_geolocation_grid(xml_path):
    """
    Parse the geolocation grid from a Sentinel-1 annotation XML.

    Returns (lines, pixels, lats, lons) where lines and pixels are 1-D arrays
    of grid coordinates, and lats/lons are 2-D arrays of shape (len(lines),
    len(pixels)).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gcp_list = root.find(".//geolocationGridPointList")
    if gcp_list is None:
        raise ValueError("No geolocationGridPointList found in XML")

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

    # Reshape into 2-D grid
    lats_grid = np.empty((n_lines, n_pixels))
    lons_grid = np.empty((n_lines, n_pixels))
    for i, ln in enumerate(unique_lines):
        for j, px in enumerate(unique_pixels):
            idx = np.where((raw_lines == ln) & (raw_pixels == px))[0][0]
            lats_grid[i, j] = raw_lats[idx]
            lons_grid[i, j] = raw_lons[idx]

    return unique_lines.astype(np.float64), unique_pixels.astype(np.float64), lats_grid, lons_grid


def build_inverse_interpolator(lines, pixels, lats, lons):
    """
    Build interpolators that map (lat, lon) -> (line, pixel).

    Returns (interp_line, interp_pixel) callables.
    """
    points = np.column_stack([lats.ravel(), lons.ravel()])
    line_vals = np.tile(lines, len(pixels))  # repeat each line for each pixel
    pixel_vals = np.repeat(pixels, len(lines))  # repeat all pixels for each line

    # Actually: for the grid, each row has the same line, each column has the same pixel
    # points[i*n_pixels + j] = (lats[i, j], lons[i, j]) -> line=lines[i], pixel=pixels[j]
    line_vals = np.repeat(lines, len(pixels))
    pixel_vals = np.tile(pixels, len(lines))

    interp_line = LinearNDInterpolator(points, line_vals, fill_value=np.nan)
    interp_pixel = LinearNDInterpolator(points, pixel_vals, fill_value=np.nan)

    return interp_line, interp_pixel


def compute_output_grid(lats, lons, pixel_size_m=10.0):
    """
    Determine the output lat/lon grid bounds and resolution.

    Returns (out_h, out_w, res_lat, res_lon, lat_min, lat_max, lon_min, lon_max).
    """
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # Add a small buffer to avoid edge effects
    lat_min -= 0.001
    lat_max += 0.001
    lon_min -= 0.001
    lon_max += 0.001

    center_lat = (lat_min + lat_max) / 2
    res_lat = pixel_size_m / M_PER_DEG_LAT
    res_lon = pixel_size_m / (M_PER_DEG_LAT * np.cos(np.radians(center_lat)))

    out_h = int(np.ceil((lat_max - lat_min) / res_lat))
    out_w = int(np.ceil((lon_max - lon_min) / res_lon))

    return out_h, out_w, res_lat, res_lon, lat_min, lat_max, lon_min, lon_max


def geocode_image(img, interp_line, interp_pixel, out_h, out_w, res_lat,
                  res_lon, lat_min, lat_max, lon_min, lon_max,
                  chunk_rows=500, order=1):
    """
    Resample an image from image coordinates to a regular lat/lon grid.
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
            & (in_lines >= 0) & (in_lines < H - 1)
            & (in_pixels >= 0) & (in_pixels < W - 1)
        )

        if not valid.any():
            continue

        valid_rows, valid_cols = np.where(valid)
        if len(valid_rows) == 0:
            continue

        coords = np.array([
            in_lines[valid_rows, valid_cols],
            in_pixels[valid_rows, valid_cols],
        ])
        sampled = map_coordinates(
            img.astype(np.float64), coords, order=order, mode="constant", cval=fill_val
        )
        geo_img[row_start:row_end][valid_rows, valid_cols] = sampled.astype(np.float32)

    return geo_img


def save_geocoded_geotiff(path, data, res_lat, res_lon, lat_min, lon_min, is_integer=False):
    """Save a geocoded array as a GeoTIFF with EPSG:4326."""
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


def find_annotation_xml(safe_dir):
    """Find the annotation XML in a SAFE directory (searches annotation/ root)."""
    matches = glob.glob(os.path.join(safe_dir, "annotation", "*vh*.xml")) + \
              glob.glob(os.path.join(safe_dir, "annotation", "*vv*.xml")) + \
              glob.glob(os.path.join(safe_dir, "annotation", "*.xml"))
    # Filter out calibration / noise subdirectories
    ann_matches = [m for m in matches if os.path.dirname(m) == os.path.join(safe_dir, "annotation")]
    if not ann_matches:
        raise FileNotFoundError(f"No annotation XML found in {safe_dir}/annotation/")
    return ann_matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Geocode Sentinel-1 GRD output images to lat/lon (EPSG:4326)"
    )
    parser.add_argument(
        "--input-dir", default="output",
        help="Directory containing output .tif files (default: output)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Geocode a single file instead of all .tif files",
    )
    parser.add_argument(
        "--safe-dir", default=None,
        help="SAFE directory for geolocation grid (default: auto-detect from data/)",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=10.0,
        help="Output pixel size in metres (default: 10)",
    )
    args = parser.parse_args()

    # --- Find SAFE directory for geolocation grid ---
    safe_dir = args.safe_dir
    if safe_dir is None:
        safe_dirs = sorted(glob.glob("data/*.SAFE"))
        if not safe_dirs:
            raise FileNotFoundError("No .SAFE directories found in data/")
        safe_dir = safe_dirs[0]
        print(f"Using geolocation grid from: {os.path.basename(safe_dir)}")

    ann_xml = find_annotation_xml(safe_dir)
    print(f"Annotation XML: {os.path.basename(ann_xml)}")

    # --- Parse geolocation grid ---
    print("Parsing geolocation grid...")
    lines, pixels, lats, lons = parse_geolocation_grid(ann_xml)
    print(f"  Grid: {len(lines)} lines x {len(pixels)} pixels = {len(lines)*len(pixels)} points")
    print(f"  Latitude:  [{lats.min():.6f}, {lats.max():.6f}]")
    print(f"  Longitude: [{lons.min():.6f}, {lons.max():.6f}]")

    # --- Build inverse interpolator ---
    print("Building inverse coordinate interpolator...")
    interp_line, interp_pixel = build_inverse_interpolator(lines, pixels, lats, lons)

    # --- Compute output grid ---
    out_h, out_w, res_lat, res_lon, lat_min, lat_max, lon_min, lon_max = \
        compute_output_grid(lats, lons, args.pixel_size)
    print(f"  Output grid: {out_w} x {out_h} pixels")
    print(f"  Resolution: {res_lat:.7f} deg lat, {res_lon:.7f} deg lon")
    print(f"  Bounds: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")

    # --- Determine input files ---
    if args.input:
        input_files = [args.input]
    else:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    print(f"\nGeocoding {len(input_files)} file(s):")

    out_dir = args.input_dir if args.input is None else os.path.dirname(args.input)
    if out_dir is None:
        out_dir = "output"

    for input_path in input_files:
        basename = os.path.basename(input_path)
        name_no_ext = os.path.splitext(basename)[0]
        output_path = os.path.join(out_dir, f"{name_no_ext}_geo.tif")

        print(f"\n  {basename} -> {name_no_ext}_geo.tif")

        with rasterio.open(input_path) as ds:
            img = ds.read(1)

        is_integer = np.issubdtype(img.dtype, np.integer) or "disaster_map" in basename or "flood_map" in basename
        order = 0 if is_integer else 1

        t0 = time.time()
        geo_img = geocode_image(
            img, interp_line, interp_pixel,
            out_h, out_w, res_lat, res_lon,
            lat_min, lat_max, lon_min, lon_max,
            order=order,
        )
        t1 = time.time()
        print(f"    Geocoded in {t1-t0:.1f}s")

        save_geocoded_geotiff(output_path, geo_img, res_lat, res_lon, lat_min, lon_min, is_integer=is_integer)
        print(f"    Saved: {output_path}")

        del img, geo_img

    print(f"\nDone! Geocoded files saved in: {out_dir}")
    print("Open in QGIS with a satellite basemap (EPSG:4326).")


if __name__ == "__main__":
    main()
