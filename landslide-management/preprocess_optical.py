#!/usr/bin/env python3
"""
Sentinel-2 and Landsat-8/9 Optical Preprocessing for Landslide Detection

Processes Bottom-of-Atmosphere surface reflectance, applies cloud/shadow masking,
computes landslide-sensitive spectral indices, and prepares normalized band stacks.

Spectral indices for landslide detection:
  NDVI  = (B8-B4)/(B8+B4)     — vegetation vigor; landslides strip vegetation → NDVI drop
  dNDVI = NDVI_post - NDVI_pre — strongest optical change indicator (<-0.25 = slide scar)
  NDWI  = (B3-B11)/(B3+B11)   — wet mudflow channels, saturated soil
  BSI   = ((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2)) — bare soil/exposed rock signature

Monsoon note: During Jun-Sep, Sentinel-2 SCL cloud masks will flag nearly
entire scenes as invalid. This is EXPECTED. The valid_mask will be near-zero
and the model will rely on SAR features instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("preprocess_optical")

# ---------------------------------------------------------------------------
# Sentinel-2 SCL class constants
# ---------------------------------------------------------------------------
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_NOT_VEGETATED = 5
SCL_WATER = 6
SCL_UNCLASSIFIED = 7
SCL_CLOUD_MED = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW_ICE = 11

# Pixels belonging to these classes are considered invalid / unusable
SCL_INVALID_CLASSES = {3, 8, 9, 10, 11}

# Sentinel-2 reflectance scale factor (integer DN → float [0, 1])
S2_SCALE_FACTOR = 10_000.0

# Landsat Collection-2 reflectance scale + offset
LS_SCALE = 0.0000275
LS_OFFSET = -0.2

# ---------------------------------------------------------------------------
# Band file name fragments
# ---------------------------------------------------------------------------
_S2_10M_BANDS = {
    "B02": "_B02_10m",
    "B03": "_B03_10m",
    "B04": "_B04_10m",
    "B08": "_B08_10m",
}
_S2_20M_BANDS = {
    "B8A": "_B8A_20m",
    "B11": "_B11_20m",
    "B12": "_B12_20m",
}
_S2_SCL = "_SCL_20m"

_LS_BANDS = {
    "SR_B2": "_SR_B2",   # Blue
    "SR_B3": "_SR_B3",   # Green
    "SR_B4": "_SR_B4",   # Red
    "SR_B5": "_SR_B5",   # NIR
    "SR_B6": "_SR_B6",   # SWIR1
    "SR_B7": "_SR_B7",   # SWIR2
    "QA_PIXEL": "_QA_PIXEL",
}


# ===========================================================================
# Helper utilities
# ===========================================================================

def _find_band_file(directory: Path, suffix: str) -> Path:
    """Return the first .tif/.jp2 that contains *suffix* in its name."""
    for ext in ("*.tif", "*.TIF", "*.jp2", "*.JP2", "*.tiff", "*.TIFF"):
        matches = list(directory.rglob(f"*{suffix}*{ext[1:]}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find a file with suffix '{suffix}' under {directory}"
    )


def _read_single_band(path: Path) -> Tuple[np.ndarray, dict]:
    """Read a single-band raster, returning (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def _upsample_to_10m(
    arr_20m: np.ndarray,
    profile_20m: dict,
    profile_10m: dict,
) -> np.ndarray:
    """Bilinear reproject of a 20 m array to the 10 m grid."""
    dst_height = profile_10m["height"]
    dst_width = profile_10m["width"]
    dst = np.empty((dst_height, dst_width), dtype=np.float32)

    reproject(
        source=arr_20m,
        destination=dst,
        src_transform=profile_20m["transform"],
        src_crs=profile_20m["crs"],
        dst_transform=profile_10m["transform"],
        dst_crs=profile_10m["crs"],
        resampling=Resampling.bilinear,
    )
    return dst


# ===========================================================================
# 1. Sentinel-2 L2A reader
# ===========================================================================

def read_sentinel2_l2a(scene_dir_or_safe: str | Path) -> dict:
    """
    Read Sentinel-2 L2A (Bottom-of-Atmosphere) surface reflectance.

    Parameters
    ----------
    scene_dir_or_safe : str or Path
        Path to the .SAFE directory or an extracted scene directory that
        contains GRANULE/ sub-folders with *_B02_10m.jp2 etc.

    Returns
    -------
    dict
        Keys: 'B02', 'B03', 'B04', 'B08' (10 m native),
              'B8A', 'B11', 'B12' (upsampled to 10 m),
              'scl' (SCL, upsampled to 10 m),
              'valid_mask' (binary float32 array, 1 = valid),
              'profile' (rasterio profile of the 10 m grid, dtype=float32).
        All reflectance arrays are float32 in [0, 1].
    """
    scene_dir = Path(scene_dir_or_safe)
    log.info("Reading Sentinel-2 L2A from %s", scene_dir)

    # -- 10 m bands -----------------------------------------------------------
    ten_m_data: Dict[str, np.ndarray] = {}
    profile_10m: Optional[dict] = None

    for band_key, suffix in _S2_10M_BANDS.items():
        fp = _find_band_file(scene_dir, suffix)
        arr, prof = _read_single_band(fp)
        arr = arr / S2_SCALE_FACTOR
        arr = np.clip(arr, 0.0, 1.0)
        ten_m_data[band_key] = arr
        if profile_10m is None:
            profile_10m = prof
        log.debug("  %s -> %s  shape=%s", band_key, fp.name, arr.shape)

    if profile_10m is None:
        raise RuntimeError("No 10 m bands found in %s" % scene_dir)

    # -- 20 m bands (read raw, then upsample) ---------------------------------
    twenty_m_data: Dict[str, np.ndarray] = {}
    for band_key, suffix in _S2_20M_BANDS.items():
        fp = _find_band_file(scene_dir, suffix)
        arr, prof_20 = _read_single_band(fp)
        arr = arr / S2_SCALE_FACTOR
        arr = np.clip(arr, 0.0, 1.0)
        arr_up = _upsample_to_10m(arr, prof_20, profile_10m)
        twenty_m_data[band_key] = arr_up
        log.debug("  %s (20→10 m) -> %s", band_key, fp.name)

    # -- SCL ------------------------------------------------------------------
    scl_fp = _find_band_file(scene_dir, _S2_SCL)
    scl_raw, prof_scl = _read_single_band(scl_fp)
    # Nearest-neighbour for SCL to preserve class labels
    scl_up = np.empty(
        (profile_10m["height"], profile_10m["width"]), dtype=np.float32
    )
    reproject(
        source=scl_raw,
        destination=scl_up,
        src_transform=prof_scl["transform"],
        src_crs=prof_scl["crs"],
        dst_transform=profile_10m["transform"],
        dst_crs=profile_10m["crs"],
        resampling=Resampling.nearest,
    )
    scl_int = scl_up.astype(np.uint8)

    valid_mask = build_sentinel2_cloud_mask(scl_int)

    # -- Assemble output ------------------------------------------------------
    out_profile = profile_10m.copy()
    out_profile.update(dtype="float32", count=1, nodata=None)

    result: dict = {**ten_m_data, **twenty_m_data, "scl": scl_int,
                    "valid_mask": valid_mask, "profile": out_profile}
    log.info(
        "Sentinel-2 scene loaded. Valid pixel fraction: %.2f%%",
        float(valid_mask.mean()) * 100,
    )
    return result


# ===========================================================================
# 2. Landsat L2 reader
# ===========================================================================

def read_landsat_l2(scene_dir: str | Path) -> dict:
    """
    Read Landsat-8/9 Collection 2, Level-2 Surface Reflectance.

    Applies USGS scale factor: reflectance = DN * 0.0000275 − 0.2, clipped [0, 1].

    Parameters
    ----------
    scene_dir : str or Path

    Returns
    -------
    dict
        Keys: 'SR_B2' … 'SR_B7' (float32, [0,1]), 'QA_PIXEL' (uint16),
              'valid_mask' (float32 binary), 'profile'.
    """
    scene_dir = Path(scene_dir)
    log.info("Reading Landsat L2 from %s", scene_dir)

    result: dict = {}
    profile_ref: Optional[dict] = None

    for band_key, suffix in _LS_BANDS.items():
        fp = _find_band_file(scene_dir, suffix)
        arr, prof = _read_single_band(fp)

        if band_key == "QA_PIXEL":
            result["QA_PIXEL"] = arr.astype(np.uint16)
        else:
            arr = arr * LS_SCALE + LS_OFFSET
            arr = np.clip(arr, 0.0, 1.0)
            result[band_key] = arr

        if profile_ref is None:
            profile_ref = prof
        log.debug("  %s -> %s  shape=%s", band_key, fp.name, arr.shape)

    if profile_ref is None:
        raise RuntimeError("No Landsat bands found in %s" % scene_dir)

    valid_mask = build_landsat_cloud_mask(result["QA_PIXEL"])
    result["valid_mask"] = valid_mask

    out_profile = profile_ref.copy()
    out_profile.update(dtype="float32", count=1, nodata=None)
    result["profile"] = out_profile

    log.info(
        "Landsat scene loaded. Valid pixel fraction: %.2f%%",
        float(valid_mask.mean()) * 100,
    )
    return result


# ===========================================================================
# 3. Cloud / QA masks
# ===========================================================================

def build_sentinel2_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """
    Build a binary valid-pixel mask from the Sentinel-2 SCL layer.

    Parameters
    ----------
    scl : np.ndarray (uint8 or int)
        Scene Classification Layer values.

    Returns
    -------
    np.ndarray (float32)
        1.0 where the pixel is valid (clear land/water/vegetation),
        0.0 where invalid (cloud shadow, medium/high cloud, cirrus, snow/ice).
    """
    invalid = np.zeros(scl.shape, dtype=bool)
    for cls in SCL_INVALID_CLASSES:
        invalid |= scl == cls
    # SCL = 0 (No Data) is also invalid
    invalid |= scl == SCL_NO_DATA
    valid_mask = (~invalid).astype(np.float32)
    return valid_mask


def build_landsat_cloud_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """
    Build a binary valid-pixel mask from the Landsat QA_PIXEL band.

    Bit positions (0-indexed from LSB):
      bit 1 – Dilated Cloud
      bit 2 – Cirrus
      bit 3 – Cloud
      bit 4 – Cloud Shadow

    Parameters
    ----------
    qa_pixel : np.ndarray (uint16)

    Returns
    -------
    np.ndarray (float32)
        1.0 = valid, 0.0 = cloud / shadow / dilated cloud / cirrus.
    """
    # Bits 1, 2, 3, 4 indicate various cloud/shadow conditions
    cloud_bits_mask = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
    qa = qa_pixel.astype(np.uint16)
    contaminated = (qa & cloud_bits_mask).astype(bool)
    # Also mark fill values (bit 0 = Fill)
    contaminated |= (qa & 1).astype(bool)
    valid_mask = (~contaminated).astype(np.float32)
    return valid_mask


# ===========================================================================
# 4. Spectral indices
# ===========================================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red + ε)

    Vegetation vigor; landslides strip vegetation → NDVI drop from ~0.6 to <0.1.

    Returns
    -------
    np.ndarray (float32), clipped to [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_dndvi(ndvi_pre: np.ndarray, ndvi_post: np.ndarray) -> np.ndarray:
    """
    Difference NDVI (post − pre).

    Strongest optical change indicator for landslide mapping.
    Typical landslide range: [-0.6, 0.0].  Values < −0.25 are strong evidence
    of vegetation loss consistent with a slide scar.

    Returns
    -------
    np.ndarray (float32).
    """
    return (ndvi_post.astype(np.float32) - ndvi_pre.astype(np.float32))


def compute_ndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index (Gao 1996 variant using SWIR1).

    NDWI = (Green − SWIR1) / (Green + SWIR1 + ε)

    Highlights wet mudflow channels and saturated soil. High NDWI after a
    landslide indicates debris-flow run-out zones.

    Returns
    -------
    np.ndarray (float32).
    """
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    ndwi = (green - swir1) / (green + swir1 + 1e-8)
    return ndwi.astype(np.float32)


def compute_bsi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
) -> np.ndarray:
    """
    Bare Soil Index.

    BSI = ((SWIR1 + Red) − (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue) + ε)

    Highlights exposed rock and bare earth in landslide scars.

    Returns
    -------
    np.ndarray (float32).
    """
    blue  = blue.astype(np.float32)
    red   = red.astype(np.float32)
    nir   = nir.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue) + 1e-8
    bsi = num / den
    return bsi.astype(np.float32)


def compute_ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Built-up Index.

    NDBI = (SWIR1 − NIR) / (SWIR1 + NIR + ε)

    Useful for separating built-up debris from natural landslide scars.

    Returns
    -------
    np.ndarray (float32).
    """
    swir1 = swir1.astype(np.float32)
    nir   = nir.astype(np.float32)
    ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)
    return ndbi.astype(np.float32)


# ===========================================================================
# 5. Normalisation
# ===========================================================================

def normalize_band(
    band: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Percentile-based normalisation to [0, 1], robust to outliers.

    Parameters
    ----------
    band : np.ndarray
    p_low, p_high : float
        Percentiles used as min/max clip bounds.
    valid_mask : optional np.ndarray (bool/float)
        If provided, percentiles are computed only over valid pixels.

    Returns
    -------
    np.ndarray (float32), values in [0, 1].
    """
    band = band.astype(np.float32)

    if valid_mask is not None:
        valid_pixels = band[valid_mask.astype(bool)]
    else:
        valid_pixels = band.ravel()

    # Guard against fully masked scenes (e.g. 100% cloud during monsoon)
    if valid_pixels.size == 0:
        log.warning("normalize_band: no valid pixels; returning zeros.")
        return np.zeros_like(band, dtype=np.float32)

    lo = float(np.nanpercentile(valid_pixels, p_low))
    hi = float(np.nanpercentile(valid_pixels, p_high))

    if hi - lo < 1e-10:
        # Constant band (rare edge case)
        return np.zeros_like(band, dtype=np.float32)

    normed = (band - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 6. Band stacks
# ===========================================================================

def build_sentinel2_stack(
    scene_dict: dict,
    compute_indices: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble a (H, W, C) numpy array from a Sentinel-2 scene dict.

    Channel order when compute_indices=True (9 channels):
      0  B02 (Blue)
      1  B03 (Green)
      2  B04 (Red)
      3  B08 (NIR broad)
      4  B11 (SWIR1)
      5  B12 (SWIR2)
      6  NDVI
      7  NDWI
      8  BSI

    Channel order when compute_indices=False (6 channels):
      0  B02 … 5  B12

    Each band is independently normalised to [0, 1] using percentile clipping.

    Parameters
    ----------
    scene_dict : dict
        Output of :func:`read_sentinel2_l2a`.
    compute_indices : bool

    Returns
    -------
    stack : np.ndarray (H, W, C), float32
    valid_mask : np.ndarray (H, W), float32
    """
    vm = scene_dict["valid_mask"]

    blue  = scene_dict["B02"]
    green = scene_dict["B03"]
    red   = scene_dict["B04"]
    nir   = scene_dict["B08"]
    swir1 = scene_dict["B11"]
    swir2 = scene_dict["B12"]

    bands_raw = [blue, green, red, nir, swir1, swir2]
    band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]

    bands_norm = [normalize_band(b, valid_mask=vm) for b in bands_raw]

    if compute_indices:
        ndvi = compute_ndvi(nir, red)
        ndwi = compute_ndwi(green, swir1)
        bsi  = compute_bsi(blue, red, nir, swir1)

        bands_norm += [
            normalize_band(ndvi, valid_mask=vm),
            normalize_band(ndwi, valid_mask=vm),
            normalize_band(bsi, valid_mask=vm),
        ]
        band_names += ["NDVI", "NDWI", "BSI"]

    stack = np.stack(bands_norm, axis=-1).astype(np.float32)  # (H, W, C)
    log.info(
        "Sentinel-2 stack: shape=%s  channels=%s",
        stack.shape,
        band_names,
    )
    return stack, vm


def build_bitemporal_dndvi(
    pre_scene_dict: dict,
    post_scene_dict: dict,
) -> np.ndarray:
    """
    Compute bitemporal dNDVI = NDVI_post − NDVI_pre.

    Only pixels valid in BOTH scenes are used; elsewhere the output is set to 0.
    The result is normalised to [0, 1] by:
      1. Clipping raw dNDVI to [−0.8, +0.8]
      2. Rescaling to [0, 1]

    Parameters
    ----------
    pre_scene_dict  : dict  (output of read_sentinel2_l2a)
    post_scene_dict : dict

    Returns
    -------
    np.ndarray (float32, [0, 1])  same spatial shape as inputs.
    """
    ndvi_pre  = compute_ndvi(pre_scene_dict["B08"],  pre_scene_dict["B04"])
    ndvi_post = compute_ndvi(post_scene_dict["B08"], post_scene_dict["B04"])

    both_valid = (
        pre_scene_dict["valid_mask"].astype(bool)
        & post_scene_dict["valid_mask"].astype(bool)
    )

    dndvi = compute_dndvi(ndvi_pre, ndvi_post)

    # Mask invalid areas
    dndvi[~both_valid] = 0.0

    # Clip and rescale: raw range [-0.8, +0.8] → [0, 1]
    dndvi_clipped = np.clip(dndvi, -0.8, 0.8)
    dndvi_norm = (dndvi_clipped + 0.8) / 1.6

    log.info(
        "dNDVI: min=%.3f  max=%.3f  both-valid fraction=%.2f%%",
        float(dndvi.min()),
        float(dndvi.max()),
        float(both_valid.mean()) * 100,
    )
    return dndvi_norm.astype(np.float32)


# ===========================================================================
# 7. Full preprocessing pipeline
# ===========================================================================

def _save_geotiff(
    array: np.ndarray,
    profile: dict,
    output_path: Path,
) -> None:
    """Write a (H, W) or (H, W, C) float32 array to a GeoTIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        count = 1
        data = array[np.newaxis, ...]   # (1, H, W)
    else:
        # (H, W, C) → (C, H, W)
        count = array.shape[2]
        data = np.moveaxis(array, -1, 0)

    write_profile = profile.copy()
    write_profile.update(
        dtype="float32",
        count=count,
        compress="lzw",
        predictor=2,
        nodata=None,
    )
    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(data.astype(np.float32))

    log.info("Saved: %s  shape=%s", output_path, data.shape)


def preprocess_sentinel2_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """
    Full Sentinel-2 preprocessing pipeline.

    Steps
    -----
    1. Read all BOA reflectance bands (10 m and 20 m upsampled to 10 m).
    2. Build SCL cloud/shadow mask.
    3. Compute spectral indices (NDVI, NDWI, BSI, NDBI).
    4. Normalise each band independently.
    5. Save individual band GeoTIFFs + multi-band stack.

    Parameters
    ----------
    scene_dir  : Path to Sentinel-2 .SAFE or extracted directory.
    output_dir : Directory to write processed outputs.

    Returns
    -------
    dict of output file paths:
      'stack'       – 9-channel band + indices stack
      'valid_mask'  – binary cloud/shadow mask
      'ndvi'        – raw (pre-normalised) NDVI
      'dndvi'       – placeholder (computed in bitemporal step)
      'ndwi'        – raw NDWI
      'bsi'         – raw BSI
      'ndbi'        – raw NDBI
    """
    scene_dir  = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_name = scene_dir.stem.replace(".SAFE", "")
    log.info("Preprocessing Sentinel-2 scene: %s", scene_name)

    # Step 1: read
    sd = read_sentinel2_l2a(scene_dir)
    profile = sd["profile"]

    # Step 2: compute indices
    ndvi = compute_ndvi(sd["B08"], sd["B04"])
    ndwi = compute_ndwi(sd["B03"], sd["B11"])
    bsi  = compute_bsi(sd["B02"], sd["B04"], sd["B08"], sd["B11"])
    ndbi = compute_ndbi(sd["B11"], sd["B08"])

    # Step 3: build normalised stack
    stack, valid_mask = build_sentinel2_stack(sd, compute_indices=True)

    # Step 4: save
    out_files: dict = {}

    def _out(name: str) -> Path:
        return output_dir / f"{scene_name}_{name}.tif"

    _save_geotiff(stack, profile, _out("stack"))
    out_files["stack"] = str(_out("stack"))

    _save_geotiff(valid_mask, profile, _out("valid_mask"))
    out_files["valid_mask"] = str(_out("valid_mask"))

    for name, arr in [("ndvi", ndvi), ("ndwi", ndwi), ("bsi", bsi), ("ndbi", ndbi)]:
        _save_geotiff(arr, profile, _out(name))
        out_files[name] = str(_out(name))

    # Save individual raw reflectance bands
    for band_key in ("B02", "B03", "B04", "B08", "B11", "B12"):
        _save_geotiff(sd[band_key], profile, _out(band_key.lower()))
        out_files[band_key.lower()] = str(_out(band_key.lower()))

    log.info("Preprocessing complete for %s. Outputs in %s", scene_name, output_dir)
    return out_files


def preprocess_bitemporal(
    pre_scene_dir: str | Path,
    post_scene_dir: str | Path,
    output_dir: str | Path,
    sensor: str = "sentinel2",
) -> dict:
    """
    Preprocess a pre/post pair and produce the dNDVI change layer.

    Parameters
    ----------
    pre_scene_dir  : Path to pre-event scene.
    post_scene_dir : Path to post-event scene.
    output_dir     : Output directory.
    sensor         : 'sentinel2' or 'landsat' (Landsat bitemporal not yet implemented).

    Returns
    -------
    dict with keys from both scenes plus 'dndvi'.
    """
    output_dir = Path(output_dir)

    if sensor.lower() in ("sentinel2", "s2"):
        pre_files  = preprocess_sentinel2_scene(pre_scene_dir,  output_dir / "pre")
        post_files = preprocess_sentinel2_scene(post_scene_dir, output_dir / "post")

        pre_sd  = read_sentinel2_l2a(pre_scene_dir)
        post_sd = read_sentinel2_l2a(post_scene_dir)

        dndvi = build_bitemporal_dndvi(pre_sd, post_sd)
        profile = pre_sd["profile"]

        dndvi_path = output_dir / "dndvi.tif"
        _save_geotiff(dndvi, profile, dndvi_path)

        return {
            "pre":   pre_files,
            "post":  post_files,
            "dndvi": str(dndvi_path),
        }
    else:
        raise NotImplementedError(
            f"Bitemporal preprocessing for sensor '{sensor}' is not yet implemented."
        )


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sentinel-2 / Landsat optical preprocessing for landslide detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess a single Sentinel-2 scene
  python preprocess_optical.py --pre S2A_MSIL2A_20240729.SAFE --sensor sentinel2 --output output/optical/pre

  # Bitemporal pre/post pair with dNDVI
  python preprocess_optical.py --pre S2A_20240729.SAFE --post S2B_20240801.SAFE \\
      --sensor sentinel2 --output output/optical/wayanad
""",
    )
    p.add_argument("--pre",    required=True, help="Pre-event scene directory / .SAFE")
    p.add_argument("--post",   default=None,  help="Post-event scene directory (optional for bitemporal)")
    p.add_argument("--sensor", default="sentinel2",
                   choices=["sentinel2", "s2", "landsat", "ls"],
                   help="Sensor type (default: sentinel2)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--no-indices", action="store_true",
                   help="Skip spectral index computation (faster, fewer channels)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)

    if args.post:
        log.info("Bitemporal mode: pre=%s  post=%s", args.pre, args.post)
        result = preprocess_bitemporal(
            pre_scene_dir=args.pre,
            post_scene_dir=args.post,
            output_dir=output_dir,
            sensor=args.sensor,
        )
        log.info("Bitemporal outputs:")
        for k, v in result.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    log.info("  [%s] %s: %s", k, kk, vv)
            else:
                log.info("  %s: %s", k, v)
    else:
        log.info("Single scene mode: scene=%s", args.pre)
        result = preprocess_sentinel2_scene(
            scene_dir=args.pre,
            output_dir=output_dir,
        )
        log.info("Outputs:")
        for k, v in result.items():
            log.info("  %s: %s", k, v)

    print("\n✓ Preprocessing complete. Results written to:", output_dir)


if __name__ == "__main__":
    main()
