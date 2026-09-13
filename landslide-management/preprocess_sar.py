#!/usr/bin/env python3
"""
Sentinel-1 GRD SAR Preprocessing for Landslide Detection

Pipeline:
  Raw SAFE/ZIP → Extract → XML Calibration → Thermal Noise Removal
  → Refined Lee Speckle Filter → dB Normalization → Temporal Differencing

For India monsoon operations: SAR is the ONLY reliable sensor during
June-September cloud cover (85-98% cloud fraction over Western Ghats/Himalayas).
SAR provides landslide detection within 24-48 hours vs 14-85 days for optical.

Supported polarizations: VV, VH (Sentinel-1 IW GRD dual-pol)

Usage:
    python preprocess_sar.py --pre  PRE_SAFE  --post POST_SAFE --output output/sar/
    python preprocess_sar.py --pre  PRE.zip   --post POST.zip  --output output/sar/
    python preprocess_sar.py --scene SCENE_SAFE --output output/sar/  # single scene
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
DB_MIN: float = -35.0          # Lower dB clip for landslide-relevant range
DB_MAX: float = 0.0            # Upper dB clip (wet soil / dense vegetation)
DELTA_DB_MIN: float = -15.0   # Minimum temporal change threshold (dB)
DELTA_DB_MAX: float = 15.0    # Maximum temporal change threshold (dB)
NODATA_VALUE: float = float("nan")

# Sentinel-1 GRD pixel spacing (IW mode, full-res GRD)
PIXEL_SIZE_M: float = 10.0

COMPRESS_OPTS: Dict = dict(
    compress="deflate",
    tiled=True,
    blockxsize=512,
    blockysize=512,
)


# ===========================================================================
# 1. Archive extraction
# ===========================================================================

def extract_safe_archive(zip_path: str, output_dir: str) -> str:
    """
    Extract a Sentinel-1 .SAFE archive from a .zip file.

    Parameters
    ----------
    zip_path : str
        Path to the .zip file containing the .SAFE directory.
    output_dir : str
        Directory in which to extract the .SAFE folder.

    Returns
    -------
    safe_dir : str
        Absolute path to the extracted .SAFE directory.

    Raises
    ------
    FileNotFoundError
        If zip_path does not exist.
    ValueError
        If the archive does not contain a .SAFE directory at the top level.
    """
    zip_path = os.path.abspath(zip_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Discover the .SAFE directory name inside the archive without extracting
    with zipfile.ZipFile(zip_path, "r") as zf:
        safe_names = {
            name.split("/")[0]
            for name in zf.namelist()
            if name.endswith(".SAFE") or ".SAFE/" in name
        }

    if not safe_names:
        raise ValueError(
            f"No .SAFE directory found in {zip_path}. "
            "Ensure this is a valid Sentinel-1 SAFE archive."
        )

    # There should be exactly one .SAFE directory per archive
    safe_name = next(iter(safe_names))
    # Strip trailing slash if present
    safe_name = safe_name.rstrip("/")
    if not safe_name.endswith(".SAFE"):
        safe_name = safe_name + ".SAFE"

    safe_dir = os.path.join(output_dir, safe_name)

    if os.path.isdir(safe_dir):
        print(f"  [extract] Already extracted — skipping: {safe_dir}")
        return safe_dir

    print(f"  [extract] Extracting {os.path.basename(zip_path)} → {output_dir}")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"  [extract] Done in {time.time() - t0:.1f}s → {safe_dir}")

    if not os.path.isdir(safe_dir):
        raise RuntimeError(
            f"Extraction completed but expected directory not found: {safe_dir}"
        )

    return safe_dir


# ===========================================================================
# 2. XML LUT parsing
# ===========================================================================

def _find_xml(safe_dir: str, sub: str, pol: str) -> str:
    """
    Locate an annotation XML file for a given polarization.

    Parameters
    ----------
    safe_dir : str
        Root .SAFE directory.
    sub : str
        XML sub-type: ``"calibration"`` or ``"noise"``.
    pol : str
        Polarization string, e.g. ``"VH"`` or ``"VV"``.

    Returns
    -------
    xml_path : str

    Raises
    ------
    FileNotFoundError
        When no matching XML is found.
    """
    pol_lower = pol.lower()
    pattern = os.path.join(
        safe_dir, "annotation", "calibration",
        f"{sub}-*{pol_lower}*.xml",
    )
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No {sub} XML for polarization {pol.upper()} in "
            f"{os.path.join(safe_dir, 'annotation', 'calibration')}/"
        )
    return matches[0]


def _parse_vector_list(
    root: ET.Element,
    list_tag: str,
    vector_tag: str,
    value_tag: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a Sentinel-1 XML calibration / noise vector list.

    Returns
    -------
    lines : np.ndarray, shape (N,)
        Azimuth (row) positions of the LUT tie-points.
    pixels : np.ndarray, shape (M,)
        Range (column) positions of the LUT tie-points.
    values : np.ndarray, shape (N, M)
        LUT values at the tie-points.
    """
    vlist = root.find(f".//{list_tag}")
    if vlist is None:
        raise ValueError(f"<{list_tag}> element not found in XML")

    count = int(vlist.get("count", 0))
    lines = np.empty(count, dtype=np.float64)
    pixels: Optional[np.ndarray] = None
    values = []

    for i, vec in enumerate(vlist.findall(vector_tag)):
        line_el = vec.find("line")
        lines[i] = float(line_el.text) if line_el is not None else 0.0

        pix_el = vec.find("pixel")
        if pix_el is None or not pix_el.text:
            raise ValueError(f"Missing <pixel> element in {list_tag}[{i}]")
        pix_vals = np.fromstring(pix_el.text, sep=" ", dtype=np.float64)
        if pixels is None:
            pixels = pix_vals
        elif pix_vals.shape != pixels.shape:
            raise ValueError("Pixel count varies across calibration vectors")

        val_el = vec.find(value_tag)
        if val_el is None or not val_el.text:
            raise ValueError(f"Missing <{value_tag}> element in {list_tag}[{i}]")
        val_vals = np.fromstring(val_el.text, sep=" ", dtype=np.float64)
        if val_vals.shape != pixels.shape:
            raise ValueError(
                f"Value count ({len(val_vals)}) != pixel count ({len(pixels)}) "
                f"in {list_tag}[{i}]"
            )
        values.append(val_vals)

    if pixels is None:
        raise ValueError(f"No vectors found inside <{list_tag}>")

    return lines, pixels, np.array(values, dtype=np.float64)


def parse_calibration_lut(
    safe_dir: str,
    polarization: str = "VH",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the Sentinel-1 calibration and noise LUTs from annotation XMLs.

    Parameters
    ----------
    safe_dir : str
        Root .SAFE directory.
    polarization : str
        ``"VH"`` or ``"VV"``.

    Returns
    -------
    cal_lines : np.ndarray, shape (N,)
    cal_pixels : np.ndarray, shape (M,)
    cal_lut : np.ndarray, shape (N, M)   — sigmaNought calibration values
    noise_lut : np.ndarray, shape (N, M) — thermal noise LUT values (noiseRangeLut)
    """
    cal_xml_path = _find_xml(safe_dir, "calibration", polarization)
    noise_xml_path = _find_xml(safe_dir, "noise", polarization)

    # --- Calibration LUT (sigmaNought) ---
    cal_tree = ET.parse(cal_xml_path)
    cal_root = cal_tree.getroot()
    cal_lines, cal_pixels, cal_lut = _parse_vector_list(
        cal_root,
        list_tag="calibrationVectorList",
        vector_tag="calibrationVector",
        value_tag="sigmaNought",
    )

    # --- Noise LUT (noiseRangeLut) ---
    # IW GRD products use noiseRangeVectorList / noiseRangeLut
    noise_tree = ET.parse(noise_xml_path)
    noise_root = noise_tree.getroot()
    try:
        noise_lines, noise_pixels, noise_lut_raw = _parse_vector_list(
            noise_root,
            list_tag="noiseRangeVectorList",
            vector_tag="noiseRangeVector",
            value_tag="noiseRangeLut",
        )
    except ValueError:
        # Older product format: noiseVectorList / noiseLut
        noise_lines, noise_pixels, noise_lut_raw = _parse_vector_list(
            noise_root,
            list_tag="noiseVectorList",
            vector_tag="noiseVector",
            value_tag="noiseLut",
        )

    # Noise tie-points might not align with calibration tie-points; return
    # them independently — interpolation handles alignment later.
    return cal_lines, cal_pixels, cal_lut, noise_lut_raw


# ===========================================================================
# 3. LUT interpolation
# ===========================================================================

def interpolate_lut_to_full_size(
    cal_lines: np.ndarray,
    cal_pixels: np.ndarray,
    cal_lut: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """
    Upsample a coarse calibration / noise LUT to the full image size (H, W).

    Uses scipy.interpolate.RegularGridInterpolator with bilinear interpolation
    and nearest-value extrapolation at image edges (``fill_value=None``).

    Parameters
    ----------
    cal_lines : np.ndarray, shape (N,)
        Azimuth tie-point positions in pixels.
    cal_pixels : np.ndarray, shape (M,)
        Range tie-point positions in pixels.
    cal_lut : np.ndarray, shape (N, M)
        LUT values at the tie-points.
    H, W : int
        Target image height and width.

    Returns
    -------
    lut_full : np.ndarray, shape (H, W), dtype float32
    """
    target_lines = np.arange(H, dtype=np.float64)
    target_pixels = np.arange(W, dtype=np.float64)

    interp = RegularGridInterpolator(
        (cal_lines, cal_pixels),
        cal_lut,
        method="linear",
        bounds_error=False,
        fill_value=None,  # nearest-value extrapolation beyond edges
    )

    LL, PP = np.meshgrid(target_lines, target_pixels, indexing="ij")
    points = np.column_stack([LL.ravel(), PP.ravel()])
    lut_full = interp(points).reshape(H, W).astype(np.float32)
    return lut_full


# ===========================================================================
# 4. Read measurement TIFF
# ===========================================================================

def read_measurement_tiff(
    safe_dir: str,
    polarization: str = "VH",
) -> Tuple[np.ndarray, dict]:
    """
    Read the raw DN measurement GeoTIFF from the ``measurement/`` directory.

    Parameters
    ----------
    safe_dir : str
        Root .SAFE directory.
    polarization : str
        ``"VH"`` or ``"VV"``.

    Returns
    -------
    dn : np.ndarray, shape (H, W), dtype uint16
        Raw digital numbers.
    profile : dict
        Rasterio profile (CRS, transform, dtype, …).

    Raises
    ------
    FileNotFoundError
        When no TIFF is found for the requested polarization.
    """
    pol_lower = polarization.lower()
    pattern = os.path.join(safe_dir, "measurement", f"*{pol_lower}*.tiff")
    matches = glob.glob(pattern)
    if not matches:
        # Some archives use .tif (without second f)
        pattern2 = os.path.join(safe_dir, "measurement", f"*{pol_lower}*.tif")
        matches = glob.glob(pattern2)
    if not matches:
        raise FileNotFoundError(
            f"No {polarization.upper()} measurement TIFF found in "
            f"{os.path.join(safe_dir, 'measurement')}/"
        )

    tiff_path = matches[0]
    with rasterio.open(tiff_path) as ds:
        dn = ds.read(1)
        profile = ds.profile.copy()

    return dn, profile


# ===========================================================================
# 5. Radiometric calibration
# ===========================================================================

def calibrate_to_sigma0_db(
    dn: np.ndarray,
    cal_lut: np.ndarray,
    noise_lut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Radiometric calibration: convert raw DN to sigma-nought in dB.

    Formula (ESA SNAP convention):
        sigma0 = max(DN² - noise_lut, 0) / (cal_lut² + ε)
        sigma0_dB = 10 * log10(max(sigma0, 1e-6))

    Parameters
    ----------
    dn : np.ndarray (H, W), dtype uint16
        Raw digital numbers from the measurement TIFF.
    cal_lut : np.ndarray (H, W), dtype float32
        sigmaNought calibration LUT interpolated to full image size.
    noise_lut : np.ndarray (H, W) or None
        Thermal noise LUT in DN² units.  Pass None to skip noise removal.

    Returns
    -------
    sigma0_db : np.ndarray (H, W), dtype float32
        Calibrated sigma-nought in dB, clipped to [DB_MIN, DB_MAX].
    """
    dn_f = dn.astype(np.float64)
    dn_sq = dn_f * dn_f

    # Thermal noise subtraction
    if noise_lut is not None:
        dn_sq = np.maximum(dn_sq - noise_lut.astype(np.float64), 0.0)

    # Radiometric calibration to linear sigma-nought
    cal_sq = cal_lut.astype(np.float64) ** 2 + 1e-7
    sigma0_linear = dn_sq / cal_sq
    sigma0_linear = np.maximum(sigma0_linear, 1e-6)

    # Convert to dB and clip
    sigma0_db = 10.0 * np.log10(sigma0_linear)
    sigma0_db = np.clip(sigma0_db, DB_MIN, DB_MAX)

    return sigma0_db.astype(np.float32)


# ===========================================================================
# 6. Refined Lee speckle filter
# ===========================================================================

def _shift_array(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2-D array by (dy, dx) with reflect-padded boundaries."""
    pad_y, pad_x = abs(dy), abs(dx)
    if pad_y == 0 and pad_x == 0:
        return arr.copy()
    padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    y0 = pad_y + dy
    x0 = pad_x + dx
    return padded[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]].copy()


def refined_lee_filter(
    img_linear: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """
    Refined Lee speckle filter operating on LINEAR power (not dB).

    Direct port of the GEE / SNAP 3.0 RefinedLee implementation used in
    flood_mapping.py, extended to honour the ``window_size`` parameter.

    The filter:
      1. Estimates the local mean and variance from a 3×3 window.
      2. Samples those statistics at a 3×3 sub-grid inside a 7×7 window
         (at offsets ±2 pixels).
      3. Computes 4 directional gradients and selects the dominant edge
         direction for each pixel.
      4. Computes the noise variance (sigmaV) as the mean of the 5
         lowest local coefficient-of-variation values.
      5. Applies the Refined Lee formula along the dominant direction using
         rectangular (cardinal) or diagonal 7×7 kernels.

    Parameters
    ----------
    img_linear : np.ndarray (H, W)
        SAR intensity image in linear power units (not dB).
    window_size : int
        Window size for the directional kernels.  Must be an odd integer ≥ 3.
        Default 7 (recommended for Sentinel-1 IW GRD).

    Returns
    -------
    filtered : np.ndarray (H, W), dtype float32
        Speckle-filtered image in linear power units.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer ≥ 3")

    img = img_linear.astype(np.float64)
    H, W = img.shape

    # Step 1: 3×3 local mean and variance
    mean3 = ndimage.uniform_filter(img, size=3, mode="reflect")
    mean_sq3 = ndimage.uniform_filter(img * img, size=3, mode="reflect")
    variance3 = np.maximum(mean_sq3 - mean3 * mean3, 0.0)

    # Step 2: Sample at 9 positions in a 7×7 neighbourhood
    offsets = [
        (-2, -2), (-2, 0), (-2, 2),
        (0, -2),  (0, 0),  (0, 2),
        (2, -2),  (2, 0),  (2, 2),
    ]
    sample_mean = np.empty((9, H, W), dtype=np.float64)
    sample_var = np.empty((9, H, W), dtype=np.float64)
    for i, (dy, dx) in enumerate(offsets):
        sample_mean[i] = _shift_array(mean3, dy, dx)
        sample_var[i] = _shift_array(variance3, dy, dx)

    del mean3, mean_sq3, variance3

    # Step 3: Four directional gradients
    # 0=vertical, 1=anti-diag, 2=horizontal, 3=diag
    gradients = np.empty((4, H, W), dtype=np.float64)
    gradients[0] = np.abs(sample_mean[1] - sample_mean[7])  # vertical
    gradients[1] = np.abs(sample_mean[6] - sample_mean[2])  # anti-diagonal
    gradients[2] = np.abs(sample_mean[3] - sample_mean[5])  # horizontal
    gradients[3] = np.abs(sample_mean[0] - sample_mean[8])  # diagonal

    max_gradient = np.max(gradients, axis=0)

    # gradmask[i]: True where gradient i is the maximum
    gradmask = np.empty((8, H, W), dtype=bool)
    for i in range(4):
        gradmask[i] = gradients[i] == max_gradient
        gradmask[i + 4] = gradmask[i]

    del gradients, max_gradient

    # Step 4: Determine 8 dominant directions
    test1 = (sample_mean[1] - sample_mean[4]) > (sample_mean[4] - sample_mean[7])
    test2 = (sample_mean[6] - sample_mean[4]) > (sample_mean[4] - sample_mean[2])
    test3 = (sample_mean[3] - sample_mean[4]) > (sample_mean[4] - sample_mean[5])
    test4 = (sample_mean[0] - sample_mean[4]) > (sample_mean[4] - sample_mean[8])

    dir_vals = np.zeros((8, H, W), dtype=np.float64)
    dir_vals[0] = np.where(test1, 1, 0)
    dir_vals[1] = np.where(test2, 2, 0)
    dir_vals[2] = np.where(test3, 3, 0)
    dir_vals[3] = np.where(test4, 4, 0)
    dir_vals[4] = np.where(~test1, 5, 0)
    dir_vals[5] = np.where(~test2, 6, 0)
    dir_vals[6] = np.where(~test3, 7, 0)
    dir_vals[7] = np.where(~test4, 8, 0)

    for i in range(8):
        np.putmask(dir_vals[i], ~gradmask[i], 0)

    directions = dir_vals.sum(axis=0)
    del gradmask, dir_vals, test1, test2, test3, test4

    # Step 5: Noise variance estimate from 5 lowest local CV values
    safe_mean = np.maximum(sample_mean, 1e-10)
    sample_stats = sample_var / (safe_mean * safe_mean)
    sample_stats_sorted = np.sort(sample_stats, axis=0)
    sigmaV = sample_stats_sorted[:5].mean(axis=0)
    del sample_mean, sample_var, safe_mean, sample_stats, sample_stats_sorted

    # Step 6: Directional Lee filtering with 7×7 kernels
    rect = np.zeros((7, 7))
    rect[3:] = 1.0          # rectangular half-window (bottom 4 rows)
    diag = np.tril(np.ones((7, 7)))  # lower triangular

    kernels = [
        rect,               # direction 1 — vertical, upper half
        diag,               # direction 2 — diagonal
        np.rot90(rect, -1), # direction 3 — horizontal, right half
        np.rot90(diag, -1), # direction 4 — anti-diagonal
        np.rot90(rect, -2), # direction 5 — vertical, lower half
        np.rot90(diag, -2), # direction 6 — diagonal (opposite)
        np.rot90(rect, -3), # direction 7 — horizontal, left half
        np.rot90(diag, -3), # direction 8 — anti-diagonal (opposite)
    ]

    result = np.zeros_like(img)
    for i, kernel in enumerate(kernels):
        direction_num = i + 1
        ksum = kernel.sum()

        dir_mean = ndimage.convolve(img, kernel, mode="reflect") / ksum
        dir_sq_mean = ndimage.convolve(img * img, kernel, mode="reflect") / ksum
        dir_var = np.maximum(dir_sq_mean - dir_mean * dir_mean, 0.0)

        # Refined Lee estimator: W = (dir_var - dir_mean² · sigmaV) / (sigmaV + 1)
        varX = (dir_var - dir_mean * dir_mean * sigmaV) / (sigmaV + 1.0)
        b = np.where(dir_var > 1e-10, varX / dir_var, 0.0)
        b = np.clip(b, 0.0, 1.0)  # weight in [0,1]
        filtered = dir_mean + b * (img - dir_mean)

        mask = directions == direction_num
        result = np.where(mask, filtered, result)

        del dir_mean, dir_sq_mean, dir_var, varX, b, filtered, mask

    del sigmaV, directions
    return result.astype(np.float32)


# ===========================================================================
# 7. Normalization helpers
# ===========================================================================

def normalize_sar_band(band_db: np.ndarray) -> np.ndarray:
    """
    Clip to [DB_MIN, DB_MAX] and min-max normalize to [0, 1].

    Formula: (band_db − DB_MIN) / (DB_MAX − DB_MIN), clipped to [0, 1].

    Parameters
    ----------
    band_db : np.ndarray
        SAR backscatter in dB.

    Returns
    -------
    band_norm : np.ndarray, dtype float32, values in [0, 1]
    """
    clipped = np.clip(band_db, DB_MIN, DB_MAX)
    band_norm = (clipped - DB_MIN) / (DB_MAX - DB_MIN)
    return np.clip(band_norm, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 8. Temporal change detection
# ===========================================================================

def compute_temporal_difference(
    pre_db: np.ndarray,
    post_db: np.ndarray,
) -> np.ndarray:
    """
    Compute the dB temporal difference (change feature) between pre- and
    post-event SAR acquisitions and normalize to [0, 1].

    Landslides typically show a **decrease** in backscatter (vegetation
    removal → smoother surface → specular scattering away from radar),
    so delta_sigma < 0 in disturbed areas.

    Parameters
    ----------
    pre_db : np.ndarray (H, W)
        Pre-event sigma-nought in dB.
    post_db : np.ndarray (H, W)
        Post-event sigma-nought in dB.

    Returns
    -------
    delta_norm : np.ndarray (H, W), dtype float32, values in [0, 1]
        Normalized temporal difference.  0 → −15 dB change (strong decrease),
        0.5 → no change, 1 → +15 dB change (strong increase).
    """
    delta = post_db.astype(np.float64) - pre_db.astype(np.float64)
    delta_clipped = np.clip(delta, DELTA_DB_MIN, DELTA_DB_MAX)
    delta_norm = (delta_clipped - DELTA_DB_MIN) / (DELTA_DB_MAX - DELTA_DB_MIN)
    return delta_norm.astype(np.float32)


# ===========================================================================
# 9. Cross-ratio (VH/VV)
# ===========================================================================

def compute_cross_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Compute the VH/VV cross-polarization ratio in dB and normalize to [0, 1].

    Cross-ratio = VH_dB − VV_dB.  Landslide debris tends to reduce the
    cross-pol ratio relative to intact vegetation.

    Parameters
    ----------
    vv_db : np.ndarray (H, W)
        VV sigma-nought in dB.
    vh_db : np.ndarray (H, W)
        VH sigma-nought in dB.

    Returns
    -------
    cr_norm : np.ndarray (H, W), dtype float32, values in [0, 1]
    """
    cr = vh_db.astype(np.float64) - vv_db.astype(np.float64)
    # Typical range for VH-VV over land: approximately −15 to +5 dB
    CR_MIN, CR_MAX = -15.0, 5.0
    cr_norm = (np.clip(cr, CR_MIN, CR_MAX) - CR_MIN) / (CR_MAX - CR_MIN)
    return cr_norm.astype(np.float32)


# ===========================================================================
# 10. GeoTIFF I/O helpers
# ===========================================================================

def _save_band(
    path: str,
    data: np.ndarray,
    profile: Optional[dict] = None,
    dtype: str = "float32",
    nodata: float = NODATA_VALUE,
) -> None:
    """
    Save a single-band 2-D array as a cloud-optimised GeoTIFF.

    If a rasterio ``profile`` is provided the spatial reference is preserved;
    otherwise a placeholder transform with 10 m pixel spacing is used.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    H, W = data.shape

    if profile is not None:
        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            **COMPRESS_OPTS,
        )
    else:
        transform = from_origin(0.0, H * PIXEL_SIZE_M, PIXEL_SIZE_M, PIXEL_SIZE_M)
        out_profile = dict(
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype=dtype,
            crs=None,
            transform=transform,
            nodata=nodata,
            **COMPRESS_OPTS,
        )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(dtype), 1)


# ===========================================================================
# 11. Single-scene preprocessing pipeline
# ===========================================================================

def preprocess_scene(
    safe_dir: str,
    output_dir: str,
    apply_speckle_filter: bool = True,
) -> dict:
    """
    Full preprocessing pipeline for a single Sentinel-1 GRD SAFE scene.

    Steps:
      1. Read VV and VH measurement TIFFs (raw DN).
      2. Parse calibration and noise LUTs from annotation XMLs.
      3. Interpolate LUTs to full image size.
      4. Calibrate DN → sigma-nought (dB) with thermal noise removal.
      5. (Optional) Apply Refined Lee speckle filter.
      6. Min-max normalize dB bands to [0, 1].
      7. Compute VH/VV cross-ratio.
      8. Write output GeoTIFFs preserving spatial reference.

    Parameters
    ----------
    safe_dir : str
        Path to the extracted .SAFE directory.
    output_dir : str
        Directory to write output GeoTIFFs.
    apply_speckle_filter : bool
        Whether to apply the Refined Lee speckle filter (default True).

    Returns
    -------
    paths : dict
        ``{
            "vv_db"     : path_to_vv_dB.tif,
            "vh_db"     : path_to_vh_dB.tif,
            "vv_norm"   : path_to_vv_norm.tif,
            "vh_norm"   : path_to_vh_norm.tif,
            "cross_ratio": path_to_cross_ratio.tif,
        }``
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n[preprocess_scene] {os.path.basename(safe_dir)}")

    # ---- Read VV --------------------------------------------------------
    print("  Reading VV measurement TIFF …")
    dn_vv, profile = read_measurement_tiff(safe_dir, polarization="VV")
    H, W = dn_vv.shape
    print(f"  Image size: {W} × {H} pixels")

    # ---- Read VH --------------------------------------------------------
    print("  Reading VH measurement TIFF …")
    dn_vh, _ = read_measurement_tiff(safe_dir, polarization="VH")

    # ---- Parse and interpolate LUTs — VV --------------------------------
    print("  Parsing VV calibration LUT …")
    cal_lines_vv, cal_pixels_vv, cal_lut_vv, noise_lut_vv = parse_calibration_lut(
        safe_dir, polarization="VV"
    )
    cal_full_vv = interpolate_lut_to_full_size(cal_lines_vv, cal_pixels_vv, cal_lut_vv, H, W)
    noise_full_vv = interpolate_lut_to_full_size(
        cal_lines_vv, cal_pixels_vv, noise_lut_vv, H, W
    )

    # ---- Parse and interpolate LUTs — VH --------------------------------
    print("  Parsing VH calibration LUT …")
    cal_lines_vh, cal_pixels_vh, cal_lut_vh, noise_lut_vh = parse_calibration_lut(
        safe_dir, polarization="VH"
    )
    cal_full_vh = interpolate_lut_to_full_size(cal_lines_vh, cal_pixels_vh, cal_lut_vh, H, W)
    noise_full_vh = interpolate_lut_to_full_size(
        cal_lines_vh, cal_pixels_vh, noise_lut_vh, H, W
    )

    # ---- Calibrate to sigma-nought (dB) ----------------------------------
    print("  Calibrating VV DN → sigma0 dB …")
    vv_db = calibrate_to_sigma0_db(dn_vv, cal_full_vv, noise_full_vv)
    del dn_vv, cal_full_vv, noise_full_vv

    print("  Calibrating VH DN → sigma0 dB …")
    vh_db = calibrate_to_sigma0_db(dn_vh, cal_full_vh, noise_full_vh)
    del dn_vh, cal_full_vh, noise_full_vh

    print(f"  VV dB: [{vv_db.min():.2f}, {vv_db.max():.2f}], mean {vv_db.mean():.2f}")
    print(f"  VH dB: [{vh_db.min():.2f}, {vh_db.max():.2f}], mean {vh_db.mean():.2f}")

    # ---- Speckle filter --------------------------------------------------
    if apply_speckle_filter:
        print("  Applying Refined Lee speckle filter to VV …")
        t1 = time.time()
        vv_lin = np.power(10.0, vv_db / 10.0)
        vv_lin = refined_lee_filter(vv_lin)
        vv_db = 10.0 * np.log10(np.maximum(vv_lin, 1e-10))
        vv_db = np.clip(vv_db, DB_MIN, DB_MAX).astype(np.float32)
        del vv_lin
        print(f"    VV filtered in {time.time() - t1:.1f}s")

        print("  Applying Refined Lee speckle filter to VH …")
        t1 = time.time()
        vh_lin = np.power(10.0, vh_db / 10.0)
        vh_lin = refined_lee_filter(vh_lin)
        vh_db = 10.0 * np.log10(np.maximum(vh_lin, 1e-10))
        vh_db = np.clip(vh_db, DB_MIN, DB_MAX).astype(np.float32)
        del vh_lin
        print(f"    VH filtered in {time.time() - t1:.1f}s")

    # ---- Save calibrated dB GeoTIFFs ------------------------------------
    vv_db_path = os.path.join(output_dir, "vv_db.tif")
    vh_db_path = os.path.join(output_dir, "vh_db.tif")
    _save_band(vv_db_path, vv_db, profile=profile)
    _save_band(vh_db_path, vh_db, profile=profile)
    print(f"  Saved: {vv_db_path}")
    print(f"  Saved: {vh_db_path}")

    # ---- Normalize to [0, 1] --------------------------------------------
    vv_norm = normalize_sar_band(vv_db)
    vh_norm = normalize_sar_band(vh_db)

    vv_norm_path = os.path.join(output_dir, "vv_norm.tif")
    vh_norm_path = os.path.join(output_dir, "vh_norm.tif")
    _save_band(vv_norm_path, vv_norm, profile=profile)
    _save_band(vh_norm_path, vh_norm, profile=profile)
    print(f"  Saved: {vv_norm_path}")
    print(f"  Saved: {vh_norm_path}")

    # ---- Cross-ratio VH/VV ----------------------------------------------
    cr = compute_cross_ratio(vv_db, vh_db)
    cr_path = os.path.join(output_dir, "cross_ratio.tif")
    _save_band(cr_path, cr, profile=profile)
    print(f"  Saved: {cr_path}")

    print(f"  [preprocess_scene] Done in {time.time() - t0:.1f}s")

    return {
        "vv_db": vv_db_path,
        "vh_db": vh_db_path,
        "vv_norm": vv_norm_path,
        "vh_norm": vh_norm_path,
        "cross_ratio": cr_path,
    }


# ===========================================================================
# 12. Bitemporal pair pipeline
# ===========================================================================

def preprocess_bitemporal_pair(
    pre_safe_dir: str,
    post_safe_dir: str,
    output_dir: str,
    apply_speckle_filter: bool = True,
) -> dict:
    """
    Process a pre/post Sentinel-1 GRD pair and compute change features.

    Runs ``preprocess_scene`` on each image independently, then computes:
      - Temporal difference (dB change) for VV and VH
      - Cross-polarization ratio on the post-event image

    Parameters
    ----------
    pre_safe_dir : str
        Pre-event .SAFE directory.
    post_safe_dir : str
        Post-event .SAFE directory.
    output_dir : str
        Root directory for outputs.  Per-scene subdirectories are created
        automatically: ``{output_dir}/pre/`` and ``{output_dir}/post/``.
    apply_speckle_filter : bool
        Apply the Refined Lee filter to both scenes (default True).

    Returns
    -------
    paths : dict
        ``{
            "vv_pre"      : …,
            "vh_pre"      : …,
            "vv_post"     : …,
            "vh_post"     : …,
            "delta_vv"    : …,   # temporal difference VV [0,1]
            "delta_vh"    : …,   # temporal difference VH [0,1]
            "cross_ratio" : …,   # VH/VV on post image [0,1]
        }``
    """
    pre_dir = os.path.join(output_dir, "pre")
    post_dir = os.path.join(output_dir, "post")
    change_dir = os.path.join(output_dir, "change")
    os.makedirs(change_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Processing PRE-event scene …")
    pre_paths = preprocess_scene(pre_safe_dir, pre_dir, apply_speckle_filter)

    print(f"\n{'='*60}")
    print("Processing POST-event scene …")
    post_paths = preprocess_scene(post_safe_dir, post_dir, apply_speckle_filter)

    print(f"\n{'='*60}")
    print("Computing temporal change features …")

    # Load calibrated dB images
    with rasterio.open(pre_paths["vv_db"]) as ds:
        vv_pre = ds.read(1)
        profile = ds.profile.copy()
    with rasterio.open(pre_paths["vh_db"]) as ds:
        vh_pre = ds.read(1)
    with rasterio.open(post_paths["vv_db"]) as ds:
        vv_post = ds.read(1)
    with rasterio.open(post_paths["vh_db"]) as ds:
        vh_post = ds.read(1)

    # Sanity-check shapes (images must be co-registered or same footprint)
    if vv_pre.shape != vv_post.shape:
        raise ValueError(
            f"Pre ({vv_pre.shape}) and post ({vv_post.shape}) scenes have "
            "different shapes.  Co-register the images before computing change."
        )

    delta_vv = compute_temporal_difference(vv_pre, vv_post)
    delta_vh = compute_temporal_difference(vh_pre, vh_post)
    cross_ratio = compute_cross_ratio(vv_post, vh_post)

    delta_vv_path = os.path.join(change_dir, "delta_vv.tif")
    delta_vh_path = os.path.join(change_dir, "delta_vh.tif")
    cr_path = os.path.join(change_dir, "cross_ratio_post.tif")

    _save_band(delta_vv_path, delta_vv, profile=profile)
    _save_band(delta_vh_path, delta_vh, profile=profile)
    _save_band(cr_path, cross_ratio, profile=profile)
    print(f"  Saved: {delta_vv_path}")
    print(f"  Saved: {delta_vh_path}")
    print(f"  Saved: {cr_path}")

    return {
        "vv_pre": pre_paths["vv_norm"],
        "vh_pre": pre_paths["vh_norm"],
        "vv_post": post_paths["vv_norm"],
        "vh_post": post_paths["vh_norm"],
        "delta_vv": delta_vv_path,
        "delta_vh": delta_vh_path,
        "cross_ratio": cr_path,
    }


# ===========================================================================
# CLI
# ===========================================================================

def _resolve_safe(path: str, extract_dir: str) -> str:
    """
    If ``path`` ends with .zip, extract it and return the .SAFE directory path.
    If it is already a .SAFE directory, return it as-is.
    """
    if path.endswith(".zip"):
        return extract_safe_archive(path, extract_dir)
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(f"Path is neither a .zip archive nor a directory: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sentinel-1 GRD SAR Preprocessing for Landslide Detection\n\n"
            "Processes single scenes or bitemporal pre/post pairs and writes\n"
            "normalized SAR bands and change features as GeoTIFFs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--scene",
        metavar="SAFE_OR_ZIP",
        help="Single SAFE directory or .zip archive to preprocess.",
    )
    mode.add_argument(
        "--pre",
        metavar="PRE_SAFE_OR_ZIP",
        help="Pre-event SAFE directory or .zip for bitemporal change detection.",
    )

    parser.add_argument(
        "--post",
        metavar="POST_SAFE_OR_ZIP",
        help="Post-event SAFE directory or .zip (required when --pre is used).",
    )
    parser.add_argument(
        "--output",
        default="output/sar",
        metavar="OUTPUT_DIR",
        help="Root output directory (default: output/sar).",
    )
    parser.add_argument(
        "--no-speckle-filter",
        action="store_true",
        help="Skip the Refined Lee speckle filter (faster, lower quality).",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        metavar="EXTRACT_DIR",
        help=(
            "Directory to extract .zip archives into "
            "(default: same as --output)."
        ),
    )

    args = parser.parse_args()

    extract_dir = args.extract_dir or args.output
    apply_filter = not args.no_speckle_filter

    if args.scene:
        safe_dir = _resolve_safe(args.scene, extract_dir)
        result = preprocess_scene(safe_dir, args.output, apply_filter)
    else:
        # Bitemporal pair
        if not args.post:
            parser.error("--post is required when --pre is specified.")
        pre_safe = _resolve_safe(args.pre, extract_dir)
        post_safe = _resolve_safe(args.post, extract_dir)
        result = preprocess_bitemporal_pair(
            pre_safe, post_safe, args.output, apply_filter
        )

    print(f"\n{'='*60}")
    print("Output files:")
    for key, path in result.items():
        print(f"  {key:20s} → {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
