"""Read a satellite scene into the 14-band Landslide4Sense (L4S) stack the
trained models expect, resampled to a common 10 m grid.

Supported inputs
----------------
* Sentinel-2 ``.SAFE`` (ESA / Copernicus, L1C or L2A) -- the primary case.
* A pre-stacked multiband GeoTIFF (bands already in L4S order).
* An HLS granule folder (NASA Earthdata HLSS30, per-band GeoTIFFs).

Band order expected by the models (L4S, 14 channels)
----------------------------------------------------
  1  B01  Coastal aerosol   (60 m)
  2  B02  Blue              (10 m)
  3  B03  Green             (10 m)
  4  B04  Red               (10 m)
  5  B05  Red edge 1        (20 m)
  6  B06  Red edge 2        (20 m)
  7  B07  Red edge 3        (20 m)
  8  B08  NIR               (10 m)
  9  B09  Water vapour      (60 m)
 10  B10  Cirrus            (60 m)   <- L1C only; absent in L2A -> zero-filled
 11  B11  SWIR 1            (20 m)
 12  B12  SWIR 2            (20 m)
 13  slope (deg, from DEM)           <- needs a DEM; else zero-filled
 14  DEM   elevation (m)             <- needs a DEM; else zero-filled

Notes
-----
* Reflectance is scaled to [0,1] (DN / 10000) for interpretability; the models
  are run with *scene standardisation* by default (``predict_scene``), so the
  absolute scale does not matter.
* B8A is intentionally NOT used (L4S uses B01-B12, not B8A).
* For best accuracy supply a DEM (``dem_path``); NASADEM (NASA Earthdata) or
  Copernicus DEM covering the scene both work.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds

# L4S Sentinel-2 bands (12) in order; slope + DEM are appended as 13,14.
L4S_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06",
                "B07", "B08", "B09", "B10", "B11", "B12"]
NODATA = 0.0  # Sentinel-2 L1C/L2A DN nodata


# --------------------------------------------------------------------------- #
# Sentinel-2 .SAFE
# --------------------------------------------------------------------------- #
def _find_band_jp2(safe: Path, band: str) -> Path | None:
    """Locate a band's JP2 inside a .SAFE (L1C flat, or L2A by resolution)."""
    img = list(safe.glob("GRANULE/*/IMG_DATA"))
    if not img:
        return None
    img = img[0]
    # L2A: IMG_DATA/R10m|R20m|R60m/*_B02_10m.jp2
    for res in ("R10m", "R20m", "R60m"):
        hits = sorted((img / res).glob(f"*_{band}_*.jp2")) if (img / res).is_dir() else []
        if hits:
            return hits[0]
    # L1C: IMG_DATA/*_B02.jp2
    hits = sorted(img.glob(f"*_{band}.jp2"))
    return hits[0] if hits else None


def _resample_to(src_path: Path, ref_profile: dict) -> np.ndarray:
    """Read a raster and resample (bilinear) onto the reference 10 m grid."""
    dst = np.zeros((ref_profile["height"], ref_profile["width"]), np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return dst


def read_safe(safe_dir: str | Path, dem_path: str | Path | None = None,
              bbox: tuple | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """Sentinel-2 .SAFE -> (stack[14,H,W] float32, profile, valid[H,W] bool).

    bbox: optional (lon_min, lat_min, lon_max, lat_max) AOI crop (WGS84).
    """
    safe = Path(safe_dir)
    b02 = _find_band_jp2(safe, "B02")
    if b02 is None:
        raise FileNotFoundError(f"No B02 JP2 under {safe} (is this a .SAFE?)")

    with rasterio.open(b02) as ref:
        profile = ref.profile.copy()
        win = None
        if bbox is not None:
            from rasterio.warp import transform_bounds
            l, b, r, t = transform_bounds("EPSG:4326", ref.crs, *bbox)
            win = from_bounds(l, b, r, t, ref.transform).round_offsets().round_lengths()
            profile.update(width=int(win.width), height=int(win.height),
                           transform=ref.window_transform(win))
        else:
            profile.update(width=ref.width, height=ref.height,
                           transform=ref.transform)
    profile = {"crs": profile["crs"], "transform": profile["transform"],
               "width": int(profile["width"]), "height": int(profile["height"])}

    bands = []
    for b in L4S_S2_BANDS:
        p = _find_band_jp2(safe, b)
        if p is None:
            print(f"  [warn] band {b} not found (L2A has no B10) -> zero-filled")
            bands.append(np.zeros((profile["height"], profile["width"]), np.float32))
        else:
            bands.append(_resample_to(p, profile) / 10000.0)  # DN -> reflectance

    slope, dem = _dem_bands(dem_path, profile)
    bands.extend([slope, dem])
    stack = np.stack(bands, 0).astype(np.float32)
    valid = _valid_from_reflectance(stack)
    print(f"  scene: {stack.shape} @10m, CRS {profile['crs']}, "
          f"valid {valid.mean()*100:.0f}%")
    return stack, profile, valid


# --------------------------------------------------------------------------- #
# Pre-stacked GeoTIFF / HLS
# --------------------------------------------------------------------------- #
def read_geotiff(path: str | Path) -> tuple[np.ndarray, dict, np.ndarray]:
    """A GeoTIFF whose bands are already in L4S order (>=12 bands)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = {"crs": src.crs, "transform": src.transform,
                   "width": src.width, "height": src.height}
    if arr.shape[0] < 14:
        pad = np.zeros((14 - arr.shape[0], arr.shape[1], arr.shape[2]), np.float32)
        arr = np.concatenate([arr, pad], 0)
        print(f"  [warn] {arr.shape[0]} bands given; padded to 14")
    if arr[:12].max() > 100:  # looks like DN
        arr[:12] /= 10000.0
    valid = _valid_from_reflectance(arr)
    return arr[:14], profile, valid


def read_hls(hls_dir: str | Path, dem_path=None,
             bbox=None) -> tuple[np.ndarray, dict, np.ndarray]:
    """NASA Earthdata HLSS30 granule folder (per-band COGs Bxx.tif) -> L4S stack.
    HLS is 30 m surface reflectance; resampled to its own grid, bands mapped."""
    d = Path(hls_dir)
    ref_p = None
    for b in L4S_S2_BANDS:
        hit = sorted(d.glob(f"*.{b}.tif")) or sorted(d.glob(f"*_{b}.tif"))
        if hit:
            with rasterio.open(hit[0]) as r:
                ref_p = {"crs": r.crs, "transform": r.transform,
                         "width": r.width, "height": r.height}
            break
    if ref_p is None:
        raise FileNotFoundError(f"No HLS band GeoTIFFs under {d}")
    bands = []
    for b in L4S_S2_BANDS:
        hit = sorted(d.glob(f"*.{b}.tif")) or sorted(d.glob(f"*_{b}.tif"))
        if hit:
            bands.append(_resample_to(Path(hit[0]), ref_p) / 10000.0)
        else:
            print(f"  [warn] HLS band {b} missing -> zero-filled")
            bands.append(np.zeros((ref_p["height"], ref_p["width"]), np.float32))
    slope, dem = _dem_bands(dem_path, ref_p)
    bands.extend([slope, dem])
    stack = np.stack(bands, 0).astype(np.float32)
    return stack, ref_p, _valid_from_reflectance(stack)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _dem_bands(dem_path, profile) -> tuple[np.ndarray, np.ndarray]:
    """(slope_deg, elevation) resampled to grid; zeros if no DEM."""
    h, w = profile["height"], profile["width"]
    if dem_path is None:
        print("  [warn] no DEM -> slope/DEM bands zero-filled "
              "(supply --dem for best accuracy)")
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    dem = _resample_to(Path(dem_path), profile)
    # 10 m pixel spacing; gradient in metres -> slope in degrees.
    gy, gx = np.gradient(dem.astype(np.float32), 10.0)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    return slope.astype(np.float32), dem.astype(np.float32)


def _valid_from_reflectance(stack: np.ndarray) -> np.ndarray:
    """Valid where the 12 optical bands are finite and not all-zero (nodata)."""
    opt = stack[:12]
    finite = np.all(np.isfinite(opt), axis=0)
    nonzero = np.any(opt > 1e-6, axis=0)
    return finite & nonzero


def read_scene(path: str | Path, dem_path=None,
               bbox=None) -> tuple[np.ndarray, dict, np.ndarray]:
    """Dispatch on input type: .SAFE dir, HLS dir, or GeoTIFF file."""
    p = Path(path)
    if p.is_dir() and (p.suffix.upper() == ".SAFE" or list(p.glob("GRANULE"))):
        return read_safe(p, dem_path, bbox)
    if p.is_dir():
        return read_hls(p, dem_path, bbox)
    return read_geotiff(p)
