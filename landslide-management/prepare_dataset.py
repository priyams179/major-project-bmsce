#!/usr/bin/env python3
"""
Dataset Preparation for Landslide Detection

Supports two modes:
  1. Landslide4Sense (L4S) mode: 3,799 pre-cut 128x128 patches with 14 channels
     (12 Sentinel-2 bands + ALOS PALSAR elevation + slope)
  2. Custom India scene mode: Full-scene co-registration, stacking, and patch extraction
     for Western Ghats/Himalayas/Northeast India events

Spatial Block Cross-Validation: Splits by geographic region (not random patches)
to prevent spatial autocorrelation leakage between train/val/test sets.

Class Imbalance: Landslides cover <1% of scene pixels. Uses class-balanced
sampling: 60% positive patches (containing landslide pixels), 40% negative.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# HDF5 is optional; warn if missing
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prepare_dataset")

# ===========================================================================
# Constants
# ===========================================================================

#: Bounding boxes (lon_min, lat_min, lon_max, lat_max) and UTM CRS for
#: each major landslide-prone region in India.
INDIA_REGIONS: Dict[str, dict] = {
    "western_ghats": {
        "bbox":    (73.0,  8.0, 77.5, 21.0),
        "utm":     "EPSG:32643",
        "description": "Western Ghats mountain chain (Karnataka, Kerala, Maharashtra)",
    },
    "himalayas": {
        "bbox":    (74.0, 28.0, 97.0, 37.0),
        "utm":     "EPSG:32644",
        "description": "Himalayan belt (Uttarakhand, HP, Jammu & Kashmir, Sikkim)",
    },
    "northeast": {
        "bbox":    (88.0, 22.0, 97.5, 29.5),
        "utm":     "EPSG:32646",
        "description": "Northeast India (Meghalaya, Nagaland, Manipur, Assam)",
    },
    "nilgiris": {
        "bbox":    (76.5, 10.5, 77.5, 12.0),
        "utm":     "EPSG:32643",
        "description": "Nilgiri Hills (Tamil Nadu / Kerala border)",
    },
    "eastern_ghats": {
        "bbox":    (79.0, 13.0, 87.0, 21.0),
        "utm":     "EPSG:32644",
        "description": "Eastern Ghats (Andhra Pradesh, Odisha)",
    },
}

# Default train / val / test region assignments
DEFAULT_TRAIN_REGIONS = {"western_ghats", "himalayas"}
DEFAULT_VAL_REGIONS   = {"northeast"}
DEFAULT_TEST_REGIONS  = {"nilgiris", "eastern_ghats"}

# Landslide4Sense expected patch dimensions
#: LEGACY PATH ONLY.  L4S is a Sentinel-2 dataset (12 S2 bands + ALOS DEM +
#: slope); it is retained for backwards compatibility but is no longer part
#: of the active training pipeline, which uses Sentinel-1 + Landsat-8/9.
L4S_PATCH_SIZE = 128
# Verified L4S layout (HF + empirical): B01..B12, slope, DEM
L4S_CHANNELS = 14
L4S_CHANNEL_NAMES: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12",
    "slope",
    "DEM",
]

# Minimum fraction of a patch that must be covered by valid (non-cloud) pixels
MIN_VALID_COVERAGE = 0.10

# ===========================================================================
# Canonical Sentinel-1 + Landsat channel manifest
# ===========================================================================
#: SAR is listed first because it is the only all-weather source.  During the
#: Southwest Monsoon the Landsat block is frequently mostly-masked, and leading
#: with SAR keeps the informative channels at low indices.
S1_CHANNELS: List[str] = [
    "vv_pre", "vh_pre",
    "vv_post", "vh_post",
    "delta_vv", "delta_vh",
    "vh_vv_ratio",
]

LANDSAT_CHANNELS: List[str] = [
    "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
    "NDVI", "NDWI", "BSI",
]

DEM_CHANNELS: List[str] = [
    "elevation", "slope", "sin_aspect", "cos_aspect",
    "profile_curv", "planform_curv", "twi", "tri",
]

#: Full 24-channel stack for the default S1 + Landsat + Copernicus DEM config.
S1_LANDSAT_DEM_CHANNELS: List[str] = (
    S1_CHANNELS + LANDSAT_CHANNELS + DEM_CHANNELS
)

#: Channel-name → RGB preview mapping.  Landsat band numbers carry the same
#: roles as the S2 bands they replaced, so B4/B3/B2 gives a true-colour
#: composite and B6/B5/B4 a false-colour one.
RGB_COMPOSITES: Dict[str, List[str]] = {
    "true_color": ["SR_B4", "SR_B3", "SR_B2"],
    "false_color": ["SR_B6", "SR_B5", "SR_B4"],
}


def channel_index(channels: List[str], name: str) -> Optional[int]:
    """
    Return the index of *name* in *channels*, or None if absent.

    Downstream code (RGB previews, per-band reporting) must look channels up by
    name.  Positional assumptions break the moment the stack order changes,
    which is exactly what swapping Sentinel-2 for Sentinel-1 does.
    """
    try:
        return channels.index(name)
    except ValueError:
        return None


def rgb_indices(
    channels: List[str],
    composite: str = "true_color",
) -> Optional[List[int]]:
    """
    Resolve an RGB composite name to channel indices.

    Returns None when the required bands are not in the stack, letting callers
    fall back to a SAR grayscale preview instead of showing the wrong bands.
    """
    names = RGB_COMPOSITES.get(composite)
    if not names:
        return None

    idx: List[Optional[int]] = [channel_index(channels, n) for n in names]
    if any(i is None for i in idx):
        log.debug(
            "Composite '%s' needs %s; not present in %s",
            composite, names, channels,
        )
        return None
    return [int(i) for i in idx]      # type: ignore[arg-type]


#: Ordered fallbacks used when the requested composite has no matching bands.
#: Landsat first because optical is what a human expects to see; SAR last
#: because a grey VV rendering is a fallback, not a true-colour view.
RGB_FALLBACK_CHAINS: List[List[str]] = [
    ["SR_B4", "SR_B3", "SR_B2"],   # Landsat true colour
    ["B04", "B03", "B02"],         # Sentinel-2 true colour (legacy L4S)
    ["SR_B6", "SR_B5", "SR_B4"],   # Landsat false colour (vegetation red)
    ["vv_post", "vv_post", "vv_post"],
    ["vv_pre", "vv_pre", "vv_pre"],
]


def resolve_rgb_channels(
    channels: Optional[List[str]], composite: str = "true_color",
) -> List[int]:
    """
    Pick three channel indices for display, by name rather than by position.

    Positional assumptions such as "the first three channels are RGB" were
    valid for the 12-band Sentinel-2 layout and are wrong for every other one.
    Under the S1-first stack, channels 0-2 are ``vv_pre``, ``vh_pre``,
    ``vv_post``, so a positional guess renders two radar bands and one
    backscatter image as if they were visible colours.

    Parameters
    ----------
    channels : list of str or None
        Channel manifest.  None means the layout is unknown.
    composite : str
        Requested composite, e.g. ``'true_color'`` or ``'false_color'``.

    Returns
    -------
    list of 3 int
        Indices into the channel axis, in R, G, B order.  Falls back to the
        first available bands when no named composite matches.
    """
    if channels:
        idx = rgb_indices(channels, composite)
        if idx:
            return idx

        for chain in RGB_FALLBACK_CHAINS:
            resolved = [channel_index(channels, name) for name in chain]
            if all(i is not None for i in resolved):
                return [int(i) for i in resolved]

        log.debug("No named RGB composite in manifest; using first 3 channels.")
        return list(range(min(3, len(channels))))

    # Unknown layout: fall back to the first three bands, which is the
    # convention for the legacy Sentinel-2 stack, but say so.  The caller is
    # expected to label this as a guess rather than true colour.
    log.warning(
        "Channel layout unknown; displaying the first 3 bands as RGB. Pass "
        "--channels, --composite-from-checkpoint or --manifest for a named "
        "composite."
    )
    return [0, 1, 2]


def to_hwc(
    image: np.ndarray, channels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Normalise a stack to (H, W, C).

    ``(C, H, W)`` and ``(H, W, C)`` are indistinguishable by shape alone once
    C is small relative to the patch, so the channel manifest decides: a stack
    whose last axis matches the manifest length is treated as channel-last, one
    whose first axis matches is treated as channel-first.  Only when the
    manifest is absent does the geometric heuristic apply.

    Parameters
    ----------
    image : np.ndarray
        (H, W), (H, W, C), or (C, H, W).
    channels : list of str, optional
        Channel manifest, used to disambiguate the layout.

    Returns
    -------
    np.ndarray (H, W, C).
    """
    if image.ndim == 2:
        return image[:, :, None]

    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2-D or 3-D stack, got shape {image.shape}."
        )

    n = len(channels) if channels else None
    if n is not None:
        if image.shape[-1] == n:
            return image
        if image.shape[0] == n:
            return np.transpose(image, (1, 2, 0))
        # Neither axis matches: leave it alone and let the caller fail loudly.
        return image

    # No manifest: the smaller axis is the channel axis.
    return image if image.shape[-1] <= image.shape[0] else np.transpose(image, (1, 2, 0))


def build_rgb_preview(
    image: np.ndarray,
    channels: Optional[List[str]] = None,
    composite: str = "true_color",
    percentiles: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Render an (H, W, C) stack as a stretched 3-channel RGB array.

    Parameters
    ----------
    image : np.ndarray
        (H, W, C) or (C, H, W) input.
    channels : list of str, optional
        Channel manifest used to pick the RGB bands by name.
    composite : str
        Preferred composite; falls back when its bands are absent.
    percentiles : 2-tuple of float
        Per-channel stretch limits.  Landsat and S1 have very different
        radiometry, so a fixed 0-1 scale renders one of them as noise.

    Returns
    -------
    np.ndarray (H, W, 3) float32 in [0, 1].
    """
    img = to_hwc(np.asarray(image, dtype=np.float32), channels)

    idx = resolve_rgb_channels(channels, composite)
    if not idx:
        raise ValueError(
            "Cannot choose display bands: the channel layout is unknown. Pass "
            "--channels or a checkpoint/dataset manifest."
        )

    # Silently swapping an out-of-range index for band 0 would draw the wrong
    # band while the caller's name-based request went unfulfilled.
    out_of_range = [i for i in idx if i >= img.shape[-1]]
    if out_of_range:
        raise ValueError(
            f"Display bands {out_of_range} are out of range for an image with "
            f"{img.shape[-1]} channel(s)."
        )

    rgb = img[:, :, idx].astype(np.float32).copy()

    # NaNs from padded GRD edges would propagate through the stretch.
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    lo_p, hi_p = percentiles
    for c in range(rgb.shape[-1]):
        channel = rgb[:, :, c]
        valid = channel[np.isfinite(channel)]
        if valid.size == 0:
            rgb[:, :, c] = 0.0
            continue
        lo = float(np.percentile(valid, lo_p))
        hi = float(np.percentile(valid, hi_p))
        rgb[:, :, c] = (
            np.clip((channel - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 0.0
        )

    return rgb

# ===========================================================================
# 1. Landslide4Sense patch loading
# ===========================================================================

def load_l4s_patch(h5_or_tif_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single Landslide4Sense (L4S) patch.

    The L4S dataset provides 128×128 patches with 14 channels:
      ch 0-11  – Sentinel-2 BOA reflectance bands B01–B12 (B10 included)
      ch 12    – Slope (degrees)
      ch 13    – DEM elevation (m)

    Mask pixel values:
      0   – No landslide (background)
      1   – Landslide
      -1  – Invalid / no data

    Parameters
    ----------
    h5_or_tif_path : Path
        Accepts .h5 (HDF5) or .tif (GeoTIFF) file.

    Returns
    -------
    image : np.ndarray (128, 128, 14)  float32
    mask  : np.ndarray (128, 128)      int8  values in {-1, 0, 1}
    """
    path = Path(h5_or_tif_path)
    suffix = path.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required to read HDF5 patches. "
                "Install with: pip install h5py"
            )
        with h5py.File(path, "r") as f:
            # L4S HDF5 structure may vary; try common key patterns
            if "img" in f and "mask" in f:
                image = np.array(f["img"],  dtype=np.float32)
                mask  = np.array(f["mask"], dtype=np.int8)
            elif "image" in f and "label" in f:
                image = np.array(f["image"], dtype=np.float32)
                mask  = np.array(f["label"], dtype=np.int8)
            else:
                keys = list(f.keys())
                raise KeyError(
                    f"Unknown HDF5 structure in {path}. Keys: {keys}"
                )

    elif suffix in (".tif", ".tiff"):
        # Assume channels-last or channels-first; detect from shape
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)  # (bands, H, W)

        if data.shape[0] == L4S_CHANNELS:
            # (14, 128, 128) → (128, 128, 14)
            image = np.moveaxis(data, 0, -1)
            mask_path = Path(str(path).replace("/image/", "/mask/"))
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Companion mask not found: {mask_path}"
                )
            with rasterio.open(mask_path) as msrc:
                mask = msrc.read(1).astype(np.int8)
        elif data.shape[0] == L4S_CHANNELS + 1:
            # Last band is the mask
            image = np.moveaxis(data[:L4S_CHANNELS], 0, -1)
            mask  = data[L4S_CHANNELS].astype(np.int8)
        else:
            raise ValueError(
                f"Unexpected number of bands {data.shape[0]} in {path}. "
                f"Expected {L4S_CHANNELS} (image) or {L4S_CHANNELS + 1} (image+mask)."
            )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Ensure spatial dimensions
    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image.shape[:2] != (L4S_PATCH_SIZE, L4S_PATCH_SIZE):
        log.warning(
            "Unexpected patch size: %s (expected %dx%d). Skipping validation.",
            image.shape, L4S_PATCH_SIZE, L4S_PATCH_SIZE,
        )

    return image, mask


# ===========================================================================
# 2. L4S dataset builder
# ===========================================================================

def build_l4s_dataset(
    l4s_dir: str | Path,
    split: str = "train",
) -> Tuple[List[Path], List[Path]]:
    """
    Scan an L4S dataset directory and return matched (image, mask) path pairs.

    Expected directory layouts (both supported):

    Layout A (separate dirs):
      l4s_dir/
        train/image/*.tif   train/mask/*.tif
        val/image/*.tif     val/mask/*.tif
        test/image/*.tif    test/mask/*.tif

    Layout B (splits.json):
      l4s_dir/
        image/*.tif   mask/*.tif
        splits.json   ({"train": [...], "val": [...], "test": [...]})

    Parameters
    ----------
    l4s_dir : Path
    split   : str  'train' | 'val' | 'test'

    Returns
    -------
    (image_paths, mask_paths) : tuple of sorted lists
    """
    l4s_dir = Path(l4s_dir)
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

    # --- Layout A: split subdirectories --------------------------------------
    split_img_dir  = l4s_dir / split / "image"
    split_mask_dir = l4s_dir / split / "mask"

    if split_img_dir.is_dir():
        img_paths  = sorted(split_img_dir.glob("*.tif"))
        mask_paths = sorted(split_mask_dir.glob("*.tif"))
        if not img_paths:
            img_paths  = sorted(split_img_dir.glob("*.h5"))
            mask_paths = sorted(split_mask_dir.glob("*.h5"))
        log.info(
            "L4S layout A | split=%s | images=%d  masks=%d",
            split, len(img_paths), len(mask_paths),
        )
        return img_paths, mask_paths

    # --- Layout B: splits.json -----------------------------------------------
    splits_json = l4s_dir / "splits.json"
    if splits_json.exists():
        with open(splits_json) as f:
            splits_data = json.load(f)
        filenames = splits_data.get(split, [])
        img_dir  = l4s_dir / "image"
        mask_dir = l4s_dir / "mask"
        img_paths  = [img_dir  / fn for fn in filenames]
        mask_paths = [mask_dir / fn for fn in filenames]
        log.info(
            "L4S layout B | split=%s | %d patches listed in splits.json",
            split, len(filenames),
        )
        return img_paths, mask_paths

    # --- Fallback: glob all, no split info -----------------------------------
    log.warning(
        "No split structure found in %s. Returning all images (no split filtering).",
        l4s_dir,
    )
    img_dir  = l4s_dir / "image"
    mask_dir = l4s_dir / "mask"
    if not img_dir.is_dir():
        img_dir = l4s_dir
        mask_dir = l4s_dir
    img_paths  = sorted(img_dir.glob("*.tif"))
    mask_paths = sorted(mask_dir.glob("*.tif"))
    return img_paths, mask_paths


# ===========================================================================
# 3. Raster alignment
# ===========================================================================

def align_to_reference(
    src_array: np.ndarray,
    src_profile: dict,
    ref_profile: dict,
    resampling: str = "bilinear",
) -> np.ndarray:
    """
    Reproject / resample *src_array* to match *ref_profile*.

    Parameters
    ----------
    src_array   : np.ndarray (H, W) or (C, H, W)
    src_profile : rasterio profile of source
    ref_profile : rasterio profile describing the target grid
    resampling  : 'bilinear' | 'nearest' | 'cubic' | 'lanczos'

    Returns
    -------
    np.ndarray with shape (ref_H, ref_W) or (C, ref_H, ref_W).
    """
    rs_map = {
        "bilinear": Resampling.bilinear,
        "nearest":  Resampling.nearest,
        "cubic":    Resampling.cubic,
        "lanczos":  Resampling.lanczos,
        "average":  Resampling.average,
    }
    rs = rs_map.get(resampling.lower(), Resampling.bilinear)

    dst_h = ref_profile["height"]
    dst_w = ref_profile["width"]

    is_2d = src_array.ndim == 2
    if is_2d:
        src_array = src_array[np.newaxis, ...]  # (1, H, W)

    n_bands = src_array.shape[0]
    dst = np.empty((n_bands, dst_h, dst_w), dtype=np.float32)

    for b in range(n_bands):
        reproject(
            source=src_array[b].astype(np.float32),
            destination=dst[b],
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=rs,
        )

    if is_2d:
        return dst[0]   # (H, W)
    return dst           # (C, H, W)


# ===========================================================================
# 4. Multi-modal band stacking
# ===========================================================================

def stack_modalities(
    layers: Sequence[Tuple[str, Optional[np.ndarray]]],
    channel_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate modality arrays along the channel axis in explicit order.

    Unlike the older optical/sar/dem keyword form, the caller controls position:
    Sentinel-1 must lead the stack because it is the only all-weather source.

    All arrays must share the same (H, W) footprint.  Arrays may be:
      - 2-D (H, W)      → treated as 1 channel
      - 3-D (H, W, C)  → multi-channel

    Parameters
    ----------
    layers : sequence of (str, ndarray or None)
        Ordered (modality_label, array) pairs.  Concatenation follows list
        order.  None entries are skipped.
    channel_names : list of str, optional
        Expected per-channel names, length = total channel count.  Checked
        against the inferred count to catch a stack that silently lost a band.

    Returns
    -------
    stack : np.ndarray (H, W, N_total), float32
    names : list of str
        Per-channel names, either the supplied ones or ``f"{label}_{i}"``
        placeholders.
    """
    if not layers:
        raise ValueError("At least one modality must be provided.")

    arrays: List[np.ndarray] = []
    labels: List[str] = []

    for label, arr in layers:
        if arr is None:
            continue
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim != 3:
            raise ValueError(
                f"Band array for '{label}' must be 2-D or 3-D, got {arr.ndim}-D."
            )
        arrays.append(arr)
        labels.append(label)

    if not arrays:
        raise ValueError("At least one non-None modality must be provided.")

    shapes = [a.shape[:2] for a in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(
            "Spatial shapes do not match across modalities: "
            + ", ".join(f"{n}: {s}" for n, s in zip(labels, shapes))
        )

    stack = np.concatenate(arrays, axis=-1)
    total = stack.shape[2]

    if channel_names is None:
        names = []
        for label, arr in zip(labels, arrays):
            names.extend(
                f"{label}_{i}" for i in range(arr.shape[2])
            )
    else:
        if len(channel_names) != total:
            raise ValueError(
                f"channel_names has {len(channel_names)} entries but the "
                f"stack has {total} channels "
                f"({', '.join(f'{n}:{a.shape[2]}' for n, a in zip(labels, arrays))})."
            )
        names = list(channel_names)

    log.info(
        "Multi-modal stack: %s → shape=%s",
        " + ".join(f"{n}({a.shape[2]}ch)" for n, a in zip(labels, arrays)),
        stack.shape,
    )
    return stack, names


def stack_multimodal_bands(
    optical_bands: Optional[np.ndarray] = None,
    sar_bands: Optional[np.ndarray] = None,
    dem_bands: Optional[np.ndarray] = None,
    indices_bands: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Backwards-compatible optical/sar/dem stacker.

    Retained for the legacy L4S and single-scene custom paths.  New code should
    call :func:`stack_modalities` so that SAR can lead the channel axis.

    Returns
    -------
    np.ndarray (H, W, N_total_channels), float32.
    """
    stack_multimodal_bands_layers: List[Tuple[str, Optional[np.ndarray]]] = [
        ("optical", optical_bands),
        ("sar",     sar_bands),
        ("dem",     dem_bands),
        ("indices", indices_bands),
    ]
    stack, _ = stack_modalities(layers)      # type: ignore[arg-type]
    return stack


# ===========================================================================
# 5. Patch extraction
# ===========================================================================

def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    patch_size: int = 256,
    stride: int = 128,
    min_landslide_fraction: float = 0.0,
) -> List[dict]:
    """
    Sliding-window patch extraction over a full-scene image.

    Parameters
    ----------
    image     : np.ndarray (H, W, C)   float32
    mask      : np.ndarray (H, W)      int  {0, 1} or {-1, 0, 1}
    valid_mask: np.ndarray (H, W)      float32 {0, 1}  (1 = valid pixel)
                If None, all pixels are treated as valid.
    patch_size: int   spatial size of each square patch (pixels)
    stride    : int   sliding window stride (pixels); stride < patch_size → overlap
    min_landslide_fraction : float
        Minimum fraction of valid pixels that must be landslide (class=1).
        Set to 0.0 to include all patches (positive AND negative).
        Set to >0.0 to filter out nearly-empty positive patches.

    Returns
    -------
    list of dict:
      {
        'image'      : np.ndarray (P, P, C)
        'mask'       : np.ndarray (P, P)
        'valid_mask' : np.ndarray (P, P)   (ones if valid_mask=None)
        'row'        : int  top-left row index in source image
        'col'        : int  top-left col index in source image
        'is_positive': bool  True if patch contains any landslide pixel
      }
    """
    H, W, C = image.shape
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.float32)

    patches: List[dict] = []
    n_skipped_coverage = 0
    n_skipped_fraction = 0

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            vm_patch = valid_mask[r:r + patch_size, c:c + patch_size]

            # Skip patches with insufficient valid pixel coverage
            coverage = float(vm_patch.mean())
            if coverage < MIN_VALID_COVERAGE:
                n_skipped_coverage += 1
                continue

            m_patch = mask[r:r + patch_size, c:c + patch_size]

            # Count landslide pixels among valid pixels
            valid_bool   = vm_patch.astype(bool)
            n_valid      = int(valid_bool.sum())
            n_landslide  = int((m_patch[valid_bool] == 1).sum())
            ls_fraction  = n_landslide / (n_valid + 1e-9)
            is_positive  = n_landslide > 0

            if min_landslide_fraction > 0.0 and ls_fraction < min_landslide_fraction:
                n_skipped_fraction += 1
                continue

            patches.append({
                "image":       image[r:r + patch_size, c:c + patch_size, :].copy(),
                "mask":        m_patch.copy(),
                "valid_mask":  vm_patch.copy(),
                "row":         r,
                "col":         c,
                "is_positive": is_positive,
                "ls_fraction": ls_fraction,
                "coverage":    coverage,
            })

    log.info(
        "Patch extraction: total=%d  skipped_coverage=%d  skipped_fraction=%d",
        len(patches), n_skipped_coverage, n_skipped_fraction,
    )
    return patches


# ===========================================================================
# 6. Balanced patch sampling
# ===========================================================================

def balanced_patch_sampling(
    all_patches: List[dict],
    positive_fraction: float = 0.6,
    max_patches: Optional[int] = None,
    random_seed: int = 42,
) -> List[dict]:
    """
    Sample patches to maintain a target positive/negative ratio.

    Landslide detection is highly imbalanced (<1% of pixels are slides).
    Over-sampling positives during training is essential to avoid the model
    predicting all-negative.

    Parameters
    ----------
    all_patches       : list from :func:`extract_patches`
    positive_fraction : float  target fraction of positive patches [0.0, 1.0]
    max_patches       : int or None  cap on total returned patches
    random_seed       : int

    Returns
    -------
    list of patch dicts, shuffled.
    """
    rng = random.Random(random_seed)

    positives = [p for p in all_patches if p["is_positive"]]
    negatives = [p for p in all_patches if not p["is_positive"]]

    log.info(
        "Before balanced sampling: positives=%d  negatives=%d",
        len(positives), len(negatives),
    )

    if not positives:
        log.warning("No positive (landslide) patches found! Returning all negatives.")
        result = negatives
    elif not negatives:
        log.warning("No negative patches found! Returning all positives.")
        result = positives
    else:
        # Determine how many of each to keep
        # If we have P positives and want fraction f, negatives = P*(1-f)/f
        n_pos = len(positives)
        n_neg_target = int(n_pos * (1.0 - positive_fraction) / positive_fraction)
        n_neg_target = min(n_neg_target, len(negatives))

        sampled_negatives = rng.sample(negatives, n_neg_target)
        result = positives + sampled_negatives

    rng.shuffle(result)

    if max_patches is not None and len(result) > max_patches:
        result = result[:max_patches]

    pos_count = sum(1 for p in result if p["is_positive"])
    log.info(
        "After balanced sampling: total=%d  positives=%d (%.1f%%)",
        len(result), pos_count, 100.0 * pos_count / max(len(result), 1),
    )
    return result


# ===========================================================================
# 7. Region assignment
# ===========================================================================

def assign_region(lat: float, lon: float) -> str:
    """
    Assign a (lat, lon) coordinate to one of the INDIA_REGIONS.

    Uses first-match on bounding boxes.  Returns 'unknown' if no region matches.

    Parameters
    ----------
    lat : float  latitude  (decimal degrees, WGS84)
    lon : float  longitude (decimal degrees, WGS84)

    Returns
    -------
    str  region name key from INDIA_REGIONS (or 'unknown').
    """
    for region_name, info in INDIA_REGIONS.items():
        lon_min, lat_min, lon_max, lat_max = info["bbox"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return region_name
    return "unknown"


# ===========================================================================
# 8. Spatial block split
# ===========================================================================

#: Default held-out events for the S1 + Landsat catalog.
#:
#: The six events cover three regions: Western Ghats (wayanad, irshalwadi),
#: Himalaya (chamoli, joshimath) and Northeast (tupul, pettimudi).  Held-out
#: events are never split across sets, so the train set is whatever remains.
#:
#: Only one event is held out for validation on purpose.  Each event yields a
#: few dozen patches, and an event with a single held-out event leaves the
#: training set too small for a 24-channel ResNet-34 to generalise; adding
#: pettimudi to validation would also leave the Northeast unrepresented in
#: training.  Validation exists to pick a threshold and an epoch, not to report
#: a final score, so a noisier signal is acceptable there.
DEFAULT_VAL_EVENTS: List[str]  = ["tupul_2022"]
DEFAULT_TEST_EVENTS: List[str] = ["irshalwadi_2023", "joshimath_2023"]

#: Events left for training under the defaults above.  Recorded in the dataset
#: manifest so a run records what it actually trained on.
DEFAULT_TRAIN_EVENTS: List[str] = [
    "wayanad_2024", "chamoli_2021", "pettimudi_2020",
]


def spatial_block_split(
    patches_with_coords: List[dict],
    val_regions: Optional[List[str]] = None,
    test_regions: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split patches by geographic region (LEGACY, region-level).

    Kept for the L4S path and single-region custom scenes.  For the
    multi-event Sentinel-1 catalog use :func:`event_level_split`, which groups
    by landslide event instead.

    Each patch dict must contain 'lat' and 'lon' keys, OR a 'region' key.
    If 'region' is absent, :func:`assign_region` is called using 'lat'/'lon'.

    Parameters
    ----------
    patches_with_coords : list of patch dicts
    val_regions  : regions for validation (default: ['northeast'])
    test_regions : regions for testing    (default: ['nilgiris', 'eastern_ghats'])

    Returns
    -------
    (train_patches, val_patches, test_patches) : 3-tuple of lists
    """
    if val_regions is None:
        val_regions  = list(DEFAULT_VAL_REGIONS)
    if test_regions is None:
        test_regions = list(DEFAULT_TEST_REGIONS)

    val_set  = set(val_regions)
    test_set = set(test_regions)

    train_patches: List[dict] = []
    val_patches:   List[dict] = []
    test_patches:  List[dict] = []

    for patch in patches_with_coords:
        region = patch.get("region")
        if region is None:
            lat = patch.get("lat")
            lon = patch.get("lon")
            if lat is None or lon is None:
                # Cannot assign region; put in train
                region = "unknown"
            else:
                region = assign_region(float(lat), float(lon))
            patch["region"] = region

        if region in test_set:
            test_patches.append(patch)
        elif region in val_set:
            val_patches.append(patch)
        else:
            train_patches.append(patch)

    log.info(
        "Spatial split (region): train=%d  val=%d  test=%d",
        len(train_patches), len(val_patches), len(test_patches),
    )
    return train_patches, val_patches, test_patches


def event_level_split(
    patches: List[dict],
    val_events: Optional[List[str]] = None,
    test_events: Optional[List[str]] = None,
    train_events: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split patches by landslide event to prevent spatial data leakage.

    Patches from one event are highly autocorrelated: adjacent windows share
    terrain, land cover, and radar geometry.  Splitting randomly across those
    windows inflates test scores badly, so whole events are held out.

    Every patch from a held-out event stays in that split, which means the
    test score measures transfer to terrain the model never saw.

    Parameters
    ----------
    patches : list of dict
        Patch records; each must carry 'event_name'.
    val_events : list of str, optional
        Event keys for validation.  Default
        :data:`DEFAULT_VAL_EVENTS`.
    test_events : list of str, optional
        Event keys for testing.  Default :data:`DEFAULT_TEST_EVENTS`.
    train_events : list of str, optional
        Event keys to force into train.  Omit to train on every event that is
        not held out.

    Returns
    -------
    (train_patches, val_patches, test_patches) : 3-tuple of lists

    Raises
    ------
    ValueError
        If the three event sets overlap, or if a split would end up empty,
        which means the caller's event names did not match anything in the data.
    """
    if val_events is None:
        val_events = list(DEFAULT_VAL_EVENTS)
    if test_events is None:
        test_events = list(DEFAULT_TEST_EVENTS)

    val_set  = set(val_events)
    test_set = set(test_events)

    for label, left, right in (
        ("val", val_set, test_set),
        ("train", set(train_events or []), val_set),
        ("train", set(train_events or []), test_set),
    ):
        overlap = left & right
        if overlap:
            raise ValueError(
                f"Events {sorted(overlap)} are assigned to both {label} and the "
                f"held-out splits. Each event belongs to exactly one split."
            )

    train_patches: List[dict] = []
    val_patches:   List[dict] = []
    test_patches:  List[dict] = []

    seen: Dict[str, int] = {}
    missing_event = 0

    for patch in patches:
        event = patch.get("event_name")
        if event is None:
            # Unlabelled patch: cannot reason about leakage, so keep it out of
            # the held-out splits rather than risk contaminating them.
            missing_event += 1
            train_patches.append(patch)
            continue

        seen[event] = seen.get(event, 0) + 1

        if event in test_set:
            test_patches.append(patch)
        elif event in val_set:
            val_patches.append(patch)
        else:
            train_patches.append(patch)

    if missing_event:
        log.warning(
            "%d patch(es) have no 'event_name' and were forced into train. "
            "They may share terrain with the held-out events.",
            missing_event,
        )

    if not test_patches:
        raise ValueError(
            f"Test split is empty. Known events: {sorted(seen)}. "
            f"Requested test events: {sorted(test_set)}."
        )
    if not val_patches:
        raise ValueError(
            f"Validation split is empty. Known events: {sorted(seen)}. "
            f"Requested val events: {sorted(val_set)}."
        )

    log.info("Patch counts by event:")
    for event in sorted(seen):
        role = (
            "TEST" if event in test_set
            else "VAL" if event in val_set
            else "train"
        )
        log.info("    %-22s %-6s %d patches", event, role, seen[event])

    log.info(
        "Event-level split: train=%d  val=%d  test=%d",
        len(train_patches), len(val_patches), len(test_patches),
    )
    return train_patches, val_patches, test_patches


# ===========================================================================
# 9. Save dataset
# ===========================================================================

def save_dataset(
    patches: List[dict],
    output_dir: str | Path,
    prefix: str = "",
    channel_names: Optional[List[str]] = None,
) -> dict:
    """
    Persist patches to disk as NumPy .npy files and write metadata JSON.

    Directory layout::

        output_dir/
            image_{i:05d}.npy      – float32 (P, P, C)
            mask_{i:05d}.npy       – int8    (P, P)
            valid_{i:05d}.npy      – float32 (P, P)
            splits.json            – metadata & per-patch info

    Parameters
    ----------
patches : list of patch dicts
    output_dir : Path
    prefix     : str  prepended to every filename (e.g. 'train_')
    channel_names : list of str, optional
        Per-channel names for the stack.  Written to splits.json so training
        and inference never have to guess the channel order.  When None, names
        are read from the first patch's 'channel_names' key.

    Returns
    -------
    dict  statistics:
      {
        'n_patches'        : int,
        'n_positive'       : int,
        'n_negative'       : int,
        'positive_fraction': float,
        'output_dir'       : str,
        'n_channels'       : int,
        'channel_names'    : list of str,
      }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if channel_names is None and patches:
        channel_names = patches[0].get("channel_names")

    meta_records = []
    n_positive = 0

    for i, patch in enumerate(patches):
        img_fname   = f"{prefix}image_{i:05d}.npy"
        mask_fname  = f"{prefix}mask_{i:05d}.npy"
        valid_fname = f"{prefix}valid_{i:05d}.npy"

        np.save(output_dir / img_fname,   patch["image"].astype(np.float32))
        np.save(output_dir / mask_fname,  patch["mask"].astype(np.int8))
        np.save(output_dir / valid_fname, patch["valid_mask"].astype(np.float32))

        if patch.get("is_positive", False):
            n_positive += 1

        meta_records.append({
            "index":        i,
            "image_file":   img_fname,
            "mask_file":    mask_fname,
            "valid_file":   valid_fname,
            "row":          patch.get("row"),
            "col":          patch.get("col"),
            "is_positive":  patch.get("is_positive", False),
            "ls_fraction":  round(float(patch.get("ls_fraction", 0.0)), 6),
            "coverage":     round(float(patch.get("coverage",    1.0)), 4),
            "region":       patch.get("region", "unknown"),
            "event_name":   patch.get("event_name"),
            "lat":          patch.get("lat"),
            "lon":          patch.get("lon"),
        })

    n_total = len(patches)
    n_channels = int(patches[0]["image"].shape[2]) if patches else 0

    splits_path = output_dir / f"{prefix}splits.json"
    with open(splits_path, "w") as f:
        json.dump(
            {
                "n_channels":    n_channels,
                "channel_names": channel_names,
                "patches":       meta_records,
            },
            f,
            indent=2,
        )

    stats = {
        "n_patches":         n_total,
        "n_positive":        n_positive,
        "n_negative":        n_total - n_positive,
        "positive_fraction": n_positive / max(n_total, 1),
        "output_dir":        str(output_dir),
        "n_channels":        n_channels,
        "channel_names":     channel_names,
    }
    log.info(
        "Saved %d patches to %s  (positive=%d  negative=%d)",
        n_total, output_dir, n_positive, n_total - n_positive,
    )
    return stats


# ===========================================================================
# 10. Full L4S pipeline
# ===========================================================================

def prepare_l4s_dataset(
    l4s_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 128,
) -> dict:
    """
    Full pipeline for the Landslide4Sense benchmark dataset.

    Reads all train/val/test patches, validates them, applies balanced sampling
    to the training split, and writes .npy files.

    Parameters
    ----------
    l4s_dir    : Root directory of the downloaded L4S dataset.
    output_dir : Where to write processed .npy files.
    patch_size : Expected patch spatial dimension (default: 128).

    Returns
    -------
    dict with train/val/test statistics.
    """
    l4s_dir    = Path(l4s_dir)
    output_dir = Path(output_dir)
    all_stats: dict = {}

    for split in ("train", "val", "test"):
        log.info("Processing L4S split: %s", split)
        img_paths, mask_paths = build_l4s_dataset(l4s_dir, split=split)

        if not img_paths:
            log.warning("No patches found for split '%s'. Skipping.", split)
            continue

        patches: List[dict] = []
        for img_p, mask_p in zip(img_paths, mask_paths):
            try:
                image, mask = load_l4s_patch(img_p)
            except Exception as exc:
                log.warning("Skipping %s: %s", img_p.name, exc)
                continue

            n_landslide = int((mask == 1).sum())
            patches.append({
                "image":       image,
                "mask":        mask,
                "valid_mask":  (mask >= 0).astype(np.float32),
                "row":         0,
                "col":         0,
                "is_positive": n_landslide > 0,
                "ls_fraction": n_landslide / (patch_size ** 2),
                "coverage":    float((mask >= 0).mean()),
                "region":      "l4s_global",
            })

        # Balanced sampling only for training split
        if split == "train":
            patches = balanced_patch_sampling(patches, positive_fraction=0.6)

        stats = save_dataset(patches, output_dir / split, prefix=f"{split}_")
        stats["split"] = split
        all_stats[split] = stats
        log.info("L4S %s done: %s", split, stats)

    return all_stats


# ===========================================================================
# 11. Custom India scene pipeline
# ===========================================================================

def prepare_custom_india_dataset(
    scene_config: dict,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
) -> dict:
    """
    Full pipeline for a custom India landslide scene.

    Parameters
    ----------
    scene_config : dict with keys:
      Required:
        'mask'      : Path to landslide mask GeoTIFF (-1 invalid, 0 negative,
                      1 landslide)
      Optional (at least one modality required):
        'sar_stack' : Path to the canonical 7-band S1 bitemporal stack written
                      by preprocess_sar (best option: band order is fixed)
        'sar_pre'   : Path to pre-event SAR GeoTIFF  (1 or more bands)
        'sar_post'  : Path to post-event/change SAR GeoTIFF
        'landsat'   : Path to Landsat stack GeoTIFF (from preprocess_optical)
        'optical'   : Path to any other optical stack GeoTIFF
        'dem'       : Path to DEM/terrain stack GeoTIFF
      Metadata (optional):
        'event_name'      : str
        'lat'             : float   scene centre latitude
        'lon'             : float   scene centre longitude
        'region'          : str     override region assignment
        'channel_names'   : list    override the inferred channel manifest
        'split_strategy'  : str     'auto' | 'event' | 'region' | 'pool'

    output_dir : Path
    patch_size : int  spatial size of extracted patches
    stride     : int  sliding window stride

    Returns
    -------
    dict with train/val/test split statistics.
    """
    output_dir = Path(output_dir)
    event_name = scene_config.get("event_name", "custom_scene")
    log.info("Preparing custom India dataset: %s", event_name)

    full_stack, stack_channel_names, mask_data, patches = _build_scene_patches(
        scene_config, patch_size=patch_size, stride=stride
    )

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    # Split by event when this is part of a multi-event catalog, otherwise by
    # region.  A single event cannot populate all three splits, so falling back
    # to event-level split would silently produce an empty test set.
    balanced = balanced_patch_sampling(patches, positive_fraction=0.6)
    train_p, val_p, test_p = _split_patches(balanced, scene_config)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    all_stats: dict = {}
    for split, split_patches in [("train", train_p), ("val", val_p), ("test", test_p)]:
        stats = save_dataset(
            split_patches,
            output_dir / split,
            prefix=f"{split}_",
            channel_names=stack_channel_names,
        )
        stats["split"] = split
        all_stats[split] = stats

    meta = {
        "event_name":    event_name,
        "region":        patches[0]["region"] if patches else "unknown",
        "patch_size":    patch_size,
        "stride":        stride,
        "n_channels":    full_stack.shape[2],
        "channel_names": stack_channel_names,
        "splits":        all_stats,
    }
    meta_path = output_dir / "scene_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Custom India dataset done: %s", meta)
    return all_stats


def _split_patches(
    balanced: List[dict], scene_config: dict,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split balanced patches for a single scene according to ``split_strategy``.

    ``auto``     event-level when the scene mixes events, else spatial blocks.
    ``event``    always event-level.
    ``region``   always spatial blocks.
    ``pool``     no split: everything goes to train, because the caller pools
                 events and splits once at the end.
    """
    strategy = scene_config.get("split_strategy", "auto")
    if strategy == "pool":
        return balanced, [], []
    if strategy == "event" or (
        strategy == "auto"
        and len({p.get("event_name") for p in balanced}) > 1
    ):
        return event_level_split(
            balanced,
            val_events=scene_config.get("val_events"),
            test_events=scene_config.get("test_events"),
        )
    return spatial_block_split(balanced)


def _build_scene_patches(
    scene_config: dict,
    patch_size: int = 256,
    stride: int = 128,
    balance: bool = True,
    positive_fraction: float = 0.6,
) -> Tuple[np.ndarray, List[str], np.ndarray, List[dict]]:
    """
    Align every modality onto the mask grid, stack, and extract patches.

    This is the single implementation of scene assembly.  Both
    :func:`prepare_custom_india_dataset` (one scene, split here) and
    :func:`prepare_event_dataset` (many scenes, split after pooling) call it, so
    there is exactly one definition of how a channel stack is built.

    Parameters
    ----------
    scene_config : dict
        Must contain ``mask``.  Any of ``sar_stack``, ``sar_pre``,
        ``sar_post``, ``landsat``, ``optical``, ``dem`` may be present.
    patch_size, stride : int
    balance : bool
        Whether to subsample the returned patches towards
        ``positive_fraction`` positives.  Multi-event runs pass False: the class
        ratio has to be set across the whole pool, once, after every event is in.
        Balancing each event first caps each event at its own positive count and
        the second pass then thins the pool again, which quietly discards hard
        negatives without any record of it.
    positive_fraction : float
        Target positive share, used only when ``balance`` is True.

    Returns
    -------
    (full_stack, channel_names, mask_data, patches) : 4-tuple
        ``full_stack`` is (H, W, C) float32, ``mask_data`` is (H, W) int8, and
        ``patches`` is the list of annotated patch records.
    """
    event_name = scene_config.get("event_name", "custom_scene")

    # Explicit channel order.  Sentinel-1 leads so the all-weather channels sit
    # at low indices; Landsat replaces Sentinel-2 in the optical slot.
    channel_names: Optional[List[str]] = scene_config.get("channel_names")

    # Determine region
    region = scene_config.get("region")
    if region is None:
        lat = scene_config.get("lat")
        lon = scene_config.get("lon")
        if lat is not None and lon is not None:
            region = assign_region(float(lat), float(lon))
        else:
            region = "unknown"
    log.info("Assigned region: %s", region)

    # ------------------------------------------------------------------
    # Load the reference mask (defines spatial grid)
    # ------------------------------------------------------------------
    mask_path = scene_config["mask"]
    with rasterio.open(mask_path) as src:
        mask_data    = src.read(1).astype(np.int8)
        ref_profile  = src.profile.copy()

    log.info("Reference mask: shape=%s  positives=%d",
             mask_data.shape, int((mask_data == 1).sum()))

    # ------------------------------------------------------------------
    # Load and align each modality
    # ------------------------------------------------------------------
    modality_arrays: Dict[str, np.ndarray] = {}

    def _load_and_align(path: str | Path, name: str, resamp: str = "bilinear") -> Optional[np.ndarray]:
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            log.warning("%s not found at %s", name, p)
            return None
        with rasterio.open(p) as src:
            data    = src.read().astype(np.float32)   # (C, H, W)
            sprof   = src.profile.copy()

        # Check if alignment needed
        same_shape = (sprof["height"] == ref_profile["height"]
                      and sprof["width"]  == ref_profile["width"])
        same_crs   = (str(sprof["crs"]) == str(ref_profile["crs"]))
        same_trans = (sprof["transform"] == ref_profile["transform"])

        if same_shape and same_crs and same_trans:
            aligned = data
        else:
            log.info("Aligning %s to reference grid …", name)
            aligned = align_to_reference(data, sprof, ref_profile, resampling=resamp)

        return np.moveaxis(aligned, 0, -1)  # (H, W, C)

    for key, rs in [
        ("sar_stack", "bilinear"),
        ("sar_pre",  "bilinear"),
        ("sar_post", "bilinear"),
        ("landsat",  "bilinear"),
        ("optical",  "bilinear"),
        ("dem",      "bilinear"),
    ]:
        path = scene_config.get(key)
        if path is None:
            continue
        arr = _load_and_align(path, key, rs)
        if arr is not None:
            modality_arrays[key] = arr

    if not modality_arrays:
        raise ValueError(
            "scene_config must include at least one of: "
            "sar_stack, sar_pre, sar_post, landsat, optical, dem"
        )

    # ------------------------------------------------------------------
    # Stack all available modalities
    # ------------------------------------------------------------------
    # ``sar_stack`` is the canonical 7-band product written by
    # preprocess_sar.write_sar_stack, so its band order is already correct.
    # The split-file fallback assumes the caller points ``sar_pre`` at a
    # pre-event stack and ``sar_post`` at the change stack, which preserves
    # the same order.
    sar_bands = modality_arrays.get("sar_stack")
    if sar_bands is None:
        sar_bands = _hstack_if_both(
            modality_arrays.get("sar_pre"),
            modality_arrays.get("sar_post"),
        )
    landsat_bands = modality_arrays.get("landsat")
    optical_bands = modality_arrays.get("optical")

    # SAR first: it is the only modality usable through monsoon cloud.
    stack_layers: List[Tuple[str, Optional[np.ndarray]]] = [
        ("sar",     sar_bands),
        ("landsat", landsat_bands),
        ("optical", optical_bands),
        ("dem",     modality_arrays.get("dem")),
    ]

    if channel_names is None:
        # Derive names for whichever blocks are actually present.
        inferred: List[str] = []
        if sar_bands is not None:
            n = sar_bands.shape[-1]
            inferred += S1_CHANNELS[:n] if n <= len(S1_CHANNELS) else \
                [f"sar_{i}" for i in range(n)]
        if landsat_bands is not None:
            n = landsat_bands.shape[-1]
            inferred += LANDSAT_CHANNELS[:n] if n <= len(LANDSAT_CHANNELS) else \
                [f"landsat_{i}" for i in range(n)]
        if optical_bands is not None:
            n = optical_bands.shape[-1]
            inferred += [f"optical_{i}" for i in range(n)]
        if modality_arrays.get("dem") is not None:
            n = modality_arrays["dem"].shape[-1]
            inferred += DEM_CHANNELS[:n] if n <= len(DEM_CHANNELS) else \
                [f"dem_{i}" for i in range(n)]
        channel_names = inferred

    full_stack, stack_channel_names = stack_modalities(
        stack_layers, channel_names=channel_names
    )

    # Build a global valid_mask: pixels where mask is not -1
    valid_mask_global = (mask_data >= 0).astype(np.float32)

    # ------------------------------------------------------------------
    # Extract patches
    # ------------------------------------------------------------------
    all_patches = extract_patches(
        image=full_stack,
        mask=mask_data,
        valid_mask=valid_mask_global,
        patch_size=patch_size,
        stride=stride,
    )

    # Annotate with region / event metadata
    for p in all_patches:
        p["region"]       = region
        p["event_name"]   = event_name
        p["lat"]          = scene_config.get("lat")
        p["lon"]          = scene_config.get("lon")
        p["channel_names"] = stack_channel_names

    if balance:
        all_patches = balanced_patch_sampling(
            all_patches, positive_fraction=positive_fraction
        )

    return full_stack, stack_channel_names, mask_data, all_patches


def _hstack_if_both(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate two (H,W,C) arrays along channel axis, or return whichever is not None."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return np.concatenate([a, b], axis=-1)


# ===========================================================================
# 12. Dataset manifest
# ===========================================================================

def read_dataset_manifest(dataset_dir: str | Path) -> dict:
    """
    Read the channel manifest of a prepared dataset directory.

    Training, inference and evaluation all need to agree on which band sits in
    which slot, and that information is written once by
    :func:`save_dataset`.  Reading it from a single place stops those three
    from each hardcoding a channel count.

    Searches, in order: ``dataset_meta.json``, then each split's
    ``<split>_splits.json``, then the legacy unprefixed ``splits.json``.

    Parameters
    ----------
    dataset_dir : str or Path
        Root dataset directory, or a single split directory.

    Returns
    -------
    dict with keys:
      ``n_channels``    int or None  (inferred from names when absent)
      ``channel_names`` list of str or None
      ``source``        str or None   (file the values came from)
    """
    root = Path(dataset_dir)

    candidates = [root / "dataset_meta.json"]
    for split in ("train", "val", "test"):
        candidates.append(root / split / f"{split}_splits.json")
        candidates.append(root / split / "splits.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read manifest %s: %s", path, exc)
            continue

        names = meta.get("channel_names")
        n_channels = meta.get("n_channels")
        if n_channels is None and names:
            n_channels = len(names)
        if n_channels is not None:
            return {
                "n_channels": int(n_channels),
                "channel_names": list(names) if names else None,
                "source": str(path),
            }

    return {"n_channels": None, "channel_names": None, "source": None}


def summarise_channels(names: Sequence[str]) -> str:
    """
    Condense a channel manifest into '7 SAR + 9 Landsat + 8 DEM'.

    A 24-channel manifest printed in full is unreadable and hides exactly the
    thing that matters when a layout is wrong: which modality each group came
    from.
    """
    groups = []
    for modality, members in (("SAR", S1_CHANNELS),
                              ("Landsat", LANDSAT_CHANNELS),
                              ("DEM", DEM_CHANNELS)):
        found = [n for n in names if n in members]
        if found:
            groups.append(f"{len(found)} {modality}")

    leftover = [n for n in names
                if n not in S1_CHANNELS + LANDSAT_CHANNELS + DEM_CHANNELS]
    if leftover:
        groups.append(f"{len(leftover)} other")

    return " + ".join(groups) if groups else f"{len(names)} unnamed"


def infer_checkpoint_channels(checkpoint: dict) -> Optional[int]:
    """
    Recover the input width of a checkpoint from its first convolution.

    Checkpoints written before the channel manifest existed still carry the
    answer in the weight tensor.  Returns None when there is nothing to read, so
    callers can report the gap instead of silently assuming 14 channels.
    """
    state = checkpoint.get("model_state_dict") or checkpoint or {}

    # A state dict preserves module registration order, so the first 4-D weight
    # is the stem convolution and its shape[1] is the input channel count.
    # ``checkpoint`` itself is used as a fallback for weights saved without the
    # {"model_state_dict": ...} wrapper.
    for value in state.values():
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 4 and isinstance(shape[1], int):
            return int(shape[1])
    return None


def discover_split_samples(split_dir: str | Path, split: str) -> List[Tuple[Path, Path, Path]]:
    """
    Find ``(image, mask, valid)`` triples in a split directory.

    Handles both naming conventions: ``image_00001.npy`` (legacy L4S) and
    ``<split>_image_00001.npy`` (current prepare_dataset output).  The prefix is
    recovered from the filename rather than assumed, so mask and valid files are
    looked up under the same prefix as the image.

    The split-prefixed layout wins when both are present, so a stale unprefixed
    copy cannot shadow fresh data.
    """
    split_path = Path(split_dir)

    def _siblings(img: Path, prefix: str) -> Tuple[Path, Path, Path]:
        """Swap the 'image' infix for 'mask'/'valid', keeping the split prefix."""
        idx = img.name[len(prefix):]        # 'train_image_00008.npy' -> 'image_00008.npy'
        idx = idx[len("image_"):]           # -> '00008.npy'
        return (
            img,
            split_path / f"{prefix}mask_{idx}",
            split_path / f"{prefix}valid_{idx}",
        )

    prefixed = sorted(split_path.glob(f"{split}_image_*.npy"))
    if prefixed:
        prefix = f"{split}_"
        return [_siblings(img, prefix) for img in prefixed]

    return [_siblings(img, "") for img in sorted(split_path.glob("image_*.npy"))]


# ===========================================================================
# 12. Landslide inventory polygon rasterisation
# ===========================================================================

def rasterize_landslide_polygons(
    polygons: List,
    reference_profile: dict,
    all_touched: bool = False,
    buffer_m: float = 0.0,
    source_crs: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterise landslide inventory polygons onto a reference raster grid.

    The inventory polygons come from public catalogues (GLOFAS, NASA landslide
    inventory, USGS) or manual QGIS digitising of pre/post optical pairs.  They
    are the ground truth for the Sentinel-1 pipeline, replacing the
    Sentinel-2-derived Landslide4Sense labels.

    Everything outside every polygon is emitted as -1 (invalid) rather than 0,
    so the loss and the metrics skip background the model was never shown.  That
    matters here: with six small events, unlabelled terrain is overwhelmingly
    negative, and treating it as real negatives teaches the model to suppress
    every landslide-like signal.

    Parameters
    ----------
    polygons : list
        Geometry objects with a ``.geometries`` attribute (i.e. a
        ``GeometryCollection`` or ``MultiPolygon``), or a single such object.
    reference_profile : dict
        Rasterio profile defining the output grid (height, width, transform, crs).
    all_touched : bool
        If True, every pixel touched by a polygon is set to 1.  Inventory
        polygons are typically generalised outlines, so this recovers the
        marginal pixels that a strict centre-in-polygon test would drop.
    buffer_m : float
        Optional positive buffer, in reference-CRS units, applied after
        reprojection.  Landslide runout deposits extend beyond the mapped crown
        scarp, so a small buffer recovers the deposit toe.  Use 0.0 to disable.
    source_crs : str, optional
        CRS of ``polygons``.  GeoJSON is WGS84 by definition; vector readers
        return the file's own CRS, which must be passed or a projected
        shapefile would be interpreted as degrees.

    Returns
    -------
    mask : np.ndarray (H, W) int8
        Values in {-1, 0, 1}: -1 outside the inventory, 0 inside the AOI but
        outside a landslide, 1 landslide.
    inside_aoi : np.ndarray (H, W) bool
        True where a polygon (or its buffer) covers the pixel at all.
    """
    from rasterio.features import rasterize

    try:
        from shapely.geometry import mapping
        from shapely.geometry import shape as shapely_shape
    except ImportError:
        raise ImportError(
            "shapely is required to rasterise landslide polygons.  "
            "Install with: pip install shapely"
        )

    H = reference_profile["height"]
    W = reference_profile["width"]
    ref_crs = reference_profile["crs"]

    # Normalise the input to a flat list of shapely geometries.
    geoms = _as_geometry_list(polygons)

    if not geoms:
        log.warning("rasterize_landslide_polygons: no polygons supplied.")
        return (
            np.full((H, W), -1, dtype=np.int8),
            np.zeros((H, W), dtype=bool),
        )

    # Reproject before buffering: buffer_m is in reference-CRS units, so
    # buffering in the source CRS would apply metres to degree coordinates.
    src_crs = source_crs or "EPSG:4326"
    projected = []
    for g in geoms:
        g_ref = shapely_shape(_reproject_geom(mapping(g), src_crs, ref_crs))
        if buffer_m > 0:
            g_ref = g_ref.buffer(buffer_m)
        projected.append(g_ref)

    shapes = [mapping(g) for g in projected]

    burned = rasterize(
        shapes,
        out_shape=(H, W),
        transform=reference_profile["transform"],
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    ).astype(bool)

    # Anything not covered by a polygon is out-of-inventory.
    mask = np.where(burned, 1, -1).astype(np.int8)

    log.info(
        "Rasterised %d polygon(s): %d landslide px of %d (%.2f%%)",
        len(shapes), int(burned.sum()), H * W, 100.0 * burned.mean(),
    )
    return mask, burned


def _as_geometry_list(polygons) -> List:
    """
    Normalise any polygon container into a flat list of shapely geometries.

    Handles a bare geometry, a ``MultiPolygon`` / ``GeometryCollection``, a
    list of any of those, and GeoJSON feature dicts.
    """
    from shapely.geometry import shape as shapely_shape

    if polygons is None:
        return []

    if hasattr(polygons, "geometries"):
        return list(polygons.geometries)
    if isinstance(polygons, dict):
        # Unwrap the common GeoJSON containers before treating it as a feature.
        kind = polygons.get("type")
        if kind == "FeatureCollection":
            return _as_geometry_list(polygons.get("features", []))
        if kind == "GeometryCollection":
            return list(polygons.get("geometries", []))
        if kind == "Feature":
            return [shapely_shape(polygons["geometry"])]
        return [shapely_shape(polygons)]
    if isinstance(polygons, (list, tuple)):
        out: List = []
        for item in polygons:
            out.extend(_as_geometry_list(item))
        return out
    return [polygons]


def _reproject_geom(geom_mapping: dict, src_crs: str, dst_crs: str) -> dict:
    """
    Reproject a GeoJSON-style geometry mapping between two CRSs.

    Delegates to pyproj, which is already a rasterio dependency, so no extra
    install is needed.
    """
    from rasterio.warp import transform_geom

    return transform_geom(src_crs, dst_crs, geom_mapping)


def add_negative_margin(
    mask: np.ndarray,
    inside_aoi: np.ndarray,
    reference_profile: dict,
    margin_m: float,
    min_px: int = 0,
) -> np.ndarray:
    """
    Convert a ring of margin just outside the inventory into valid negatives.

    With no margin, every pixel outside a polygon is -1 (invalid), so the model
    only ever sees positives.  A band of genuine negatives immediately around
    the landslide is what teaches it the difference between a fresh scar and
    adjacent stable slope, and it is also where false positives appear.

    The margin must be at least one patch wide or no negative patch can reach
    the coverage threshold, which silently yields an all-positive dataset.
    Callers pass ``min_px = patch_size`` to guarantee that; the geometrically
    requested width is respected when it is already larger.

    Parameters
    ----------
    mask : np.ndarray (H, W) int8
        Output of :func:`rasterize_landslide_polygons`.
    inside_aoi : np.ndarray (H, W) bool
        Where the inventory polygons cover the grid.
    reference_profile : dict
        Rasterio profile of the grid.
    margin_m : float
        Margin width in metres.  0.0 means "use min_px alone".
    min_px : int
        Lower bound on the margin in pixels.

    Returns
    -------
    np.ndarray (H, W) int8 with the margin set to 0.
    """
    px_x, px_y = _margin_pixels(
        margin_m, reference_profile,
        binary_size=max(mask.shape) * 4,
    )
    px_x = max(px_x, min_px)
    px_y = max(px_y, min_px)

    if px_x <= 0 and px_y <= 0:
        return mask

    # A binary dilation of the AOI gives the margin ring; the band between the
    # original AOI and the dilated AOI becomes valid negatives.  Dilation is one
    # numpy operation rather than a per-pixel window scan, which matters because
    # a scan over a wide margin on a full Landsat scene is O(H*W*r^2).
    try:
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * px_y + 1, 2 * px_x + 1), dtype=bool)
        dilated = binary_dilation(inside_aoi, structure=structure)
    except ImportError:
        dilated = _dilate_box(inside_aoi, px_y, px_x)

    out = mask.copy()
    ring = dilated & ~inside_aoi
    out[ring] = 0

    log.info(
        "Negative margin: %d x %d px ring -> %d valid negatives",
        px_x, px_y, int(ring.sum()),
    )
    return out


def _margin_pixels(
    margin_m: float, profile: dict, binary_size: int,
) -> Tuple[int, int]:
    """
    Convert a metric margin into a pixel count for this grid.

    Geographic CRSs report their transform in degrees, so a margin in metres has
    to be scaled by the local metres-per-degree, which shrinks with latitude.
    Ignoring that turns 120 m into millions of pixels on a degree grid.

    Returns
    -------
    (px_x, px_y) : 2-tuple of int
    """
    transform = profile["transform"]
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    crs = profile.get("crs")
    is_geographic = False
    if crs is not None:
        try:
            # is_geographic first: CRS.linear_units reports "unknown" for
            # EPSG:4326 even though it is unambiguously degrees.
            is_geographic = bool(crs.is_geographic)
        except Exception:
            is_geographic = "degree" in str(getattr(crs, "linear_units", "")).lower()

    if is_geographic:
        # Local scale factor for longitude/latitude degrees at the raster top.
        top = transform.f
        lat = float(np.clip(abs(top) if abs(top) <= 90.0 else 0.0, -89.0, 89.0))
        m_per_deg_x = 111_320.0 * float(np.cos(np.deg2rad(lat)))
        m_per_deg_y = 110_574.0
        res_m_x = res_x * m_per_deg_x
        res_m_y = res_y * m_per_deg_y
    else:
        res_m_x, res_m_y = res_x, res_y

    px_x = int(margin_m / max(res_m_x, 1e-9))
    px_y = int(margin_m / max(res_m_y, 1e-9))

    # A margin wider than the scene is meaningless and would allocate a
    # structuring element larger than the raster.
    if px_x > binary_size or px_y > binary_size:
        raise ValueError(
            f"Negative margin of {margin_m:.0f} m spans {px_x} x {px_y} px, which "
            f"exceeds the {binary_size} px grid. Reduce --negative-margin or supply "
            f"a finer-resolution reference."
        )

    return px_x, px_y


def _dilate_box(binary: np.ndarray, px_y: int, px_x: int) -> np.ndarray:
    """
    Dilate a boolean array with a rectangular structuring element.

    NumPy-only fallback for environments without SciPy: the rectangle is
    separable, so two 1-D passes give the same result as the full 2-D element.
    """
    out = binary.copy()
    for axis, half in ((0, px_y), (1, px_x)):
        if half <= 0:
            continue
        shifted = out.copy()
        for step in range(1, half + 1):
            pos = [slice(None)] * 2
            neg = [slice(None)] * 2
            pos[axis] = slice(step, None)
            neg[axis] = slice(None, -step)
            shifted[tuple(pos)] |= out[tuple(neg)]
            shifted[tuple(neg)] |= out[tuple(pos)]
        out = shifted
    return out


# ===========================================================================
# 13. Event catalog pipeline (Sentinel-1 + Landsat + DEM)
# ===========================================================================

def load_event_polygons(
    polygon_path: str | Path,
) -> Tuple[List, Optional[str]]:
    """
    Load landslide inventory polygons from GeoJSON or a vector file.

    Accepts GeoJSON (``FeatureCollection``, ``Feature``, ``GeometryCollection``,
    or a bare geometry) and anything Fiona/GDAL can open, including Shapefile
    and GeoPackage.

    Parameters
    ----------
    polygon_path : str or Path

    Returns
    -------
    (geometries, source_crs) : tuple
        ``geometries`` is a list of shapely geometries in ``source_crs``, which
        is None for GeoJSON and therefore assumed to be WGS84.

        The CRS is returned rather than dropped because a shapefile in a
        projected CRS would otherwise be rasterised as if it were degrees and
        land somewhere on the wrong continent, silently producing a mask of
        empty pixels.
    """
    path = Path(polygon_path)
    if not path.exists():
        raise FileNotFoundError(f"Polygon file not found: {path}")

    if path.suffix.lower() in (".geojson", ".json"):
        with open(path) as fh:
            payload = json.load(fh)
        return _geoms_from_geojson(payload), None

    # Any other vector format: prefer fiona, fall back to OGR.
    try:
        import fiona
        from shapely.geometry import shape as shapely_shape

        with fiona.open(path) as src:
            geoms = [shapely_shape(f["geometry"]) for f in src if f.get("geometry")]
            source_crs = src.crs_wkt or (src.crs.to_string() if src.crs else None)
        log.info("Loaded %d polygon(s) from %s via fiona", len(geoms), path.name)
        return geoms, source_crs
    except ImportError:
        pass

    try:
        from osgeo import ogr
        from shapely import wkb as shapely_wkb

        ogr.UseExceptions()
        ds = ogr.Open(str(path))
        if ds is None:
            raise RuntimeError(f"Could not open {path} with OGR.")
        layer = ds.GetLayer(0)

        # OGR geometries are not shapely objects, and the code downstream calls
        # .buffer() and .is_empty() on them, so convert through WKB here.
        geoms = [
            shapely_wkb.loads(bytes(feat.GetGeometryRef().ExportToWkb()))
            for feat in layer
            if feat.GetGeometryRef() is not None
        ]

        spatial_ref = layer.GetSpatialRef()
        source_crs = spatial_ref.ExportToWkt() if spatial_ref else None
        ds = None
        log.info("Loaded %d polygon(s) from %s via OGR", len(geoms), path.name)
        return geoms, source_crs
    except ImportError:
        raise ImportError(
            "Reading vector data needs fiona or GDAL's OGR.  Install with:\n"
            "    pip install fiona        (or)  conda install -c conda-forge gdal\n"
            "Alternatively supply a GeoJSON file, which needs neither."
        )


def _geoms_from_geojson(payload) -> List:
    """Flatten any GeoJSON container into a list of shapely geometries."""
    from shapely.geometry import shape as shapely_shape

    kind = payload.get("type")
    if kind == "FeatureCollection":
        items = [f["geometry"] for f in payload.get("features", [])]
    elif kind == "GeometryCollection":
        items = payload.get("geometries", [])
    else:
        items = [payload]

    return [shapely_shape(g) for g in items if g]


def prepare_event_dataset(
    scenes: List[dict],
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    positive_fraction: float = 0.6,
    val_events: Optional[List[str]] = None,
    test_events: Optional[List[str]] = None,
    train_events: Optional[List[str]] = None,
    negative_margin_m: float = 150.0,
    polygon_buffer_m: float = 0.0,
) -> dict:
    """
    Build a combined patch dataset from several landslide events.

    Each entry in *scenes* is a per-event config dict accepted by
    :func:`prepare_custom_india_dataset`, plus two extra keys:

      ``polygon_file`` : path to landslide inventory polygons
      ``negative_margin_m`` : optional per-event override

    Patches from all events are pooled and then split by
    :func:`event_level_split`, so the model must generalise across terrain
    rather than memorising one hillside.

    Parameters
    ----------
    scenes : list of dict
        Per-event scene configs.
    output_dir : str or Path
    patch_size, stride : int
        Patch geometry.
    positive_fraction : float
        Target fraction of positive patches after sampling.
    val_events, test_events : list of str, optional
        Held-out event keys.
    negative_margin_m : float
        Width of the valid-negative ring around each inventory polygon.
    polygon_buffer_m : float
        Optional buffer applied to polygons before rasterisation.

    Returns
    -------
    dict with per-split statistics and the channel manifest.
    """
    if not scenes:
        raise ValueError("prepare_event_dataset: no scenes supplied.")

    # Catch a typo'd event name now rather than after hours of preprocessing.
    known = {s.get("event_name") for s in scenes}
    if None in known:
        log.warning("Some scenes have no 'event_name'; they will be named automatically.")
    for label, requested in (
        ("val", val_events or DEFAULT_VAL_EVENTS),
        ("test", test_events or DEFAULT_TEST_EVENTS),
        ("train", train_events or []),
    ):
        absent = set(requested) - known
        if absent:
            raise ValueError(
                f"{label} event(s) {sorted(absent)} are not in the scene list. "
                f"Available: {sorted(n for n in known if n)}."
            )

    output_dir = Path(output_dir)
    pooled: List[dict] = []
    channel_names: Optional[List[str]] = None
    per_event: Dict[str, dict] = {}

    for i, scene in enumerate(scenes):
        event = scene.get("event_name", f"event_{i:04d}")
        log.info(
            "\n%s\n[%d/%d] %s\n%s",
            "=" * 70, i + 1, len(scenes), event, "=" * 70,
        )

        # Each event is rasterised, stacked and cut into patches on its own;
        # the split happens once at the end.  Splitting per event would be
        # impossible anyway, since one event cannot fill three splits.
        # The event name must reach the patch builder too: patches carry it, and
        # the split below routes on that field.  An unnamed scene left as None
        # would silently pool into 'train' no matter what was requested.
        scene_cfg = dict(scene)
        scene_cfg["event_name"] = event
        cfg = _materialise_event_mask(
            scene_cfg,
            output_dir / "_per_event" / event,
            negative_margin_m=scene.get("negative_margin_m", negative_margin_m),
            polygon_buffer_m=scene.get("polygon_buffer_m", polygon_buffer_m),
            patch_size=patch_size,
        )
        _, event_channels, _, patches = _build_scene_patches(
            cfg, patch_size=patch_size, stride=stride, balance=False
        )

        if channel_names is None:
            channel_names = event_channels
        elif event_channels != channel_names:
            raise ValueError(
                f"Event '{event}' produced channels {event_channels} "
                f"but earlier events produced {channel_names}. All events must "
                f"use the same channel layout."
            )

        per_event[event] = {
            "n_patches":     len(patches),
            "n_positive":    sum(1 for p in patches if p.get("is_positive")),
            "channel_names": event_channels,
        }
        pooled.extend(patches)

    # ---- Split across the whole pool -------------------------------------
    log.info(
        "\n%s\nPooled %d patches from %d events\n%s",
        "=" * 70, len(pooled), len(scenes), "=" * 70,
    )

    # Split first, then balance each split.  Balancing the pooled set would
    # subsample negatives without regard to which event they belong to, so a
    # held-out event could lose the negatives that make its score meaningful
    # while keeping every positive.
    train_p, val_p, test_p = event_level_split(
        pooled,
        val_events=val_events,
        test_events=test_events,
        train_events=train_events,
    )

    all_stats: dict = {}
    for split, patches in [("train", train_p), ("val", val_p), ("test", test_p)]:
        balanced = balanced_patch_sampling(
            patches, positive_fraction=positive_fraction
        )
        all_stats[split] = save_dataset(
            balanced,
            output_dir / split,
            prefix=f"{split}_",
            channel_names=channel_names,
        )
        all_stats[split]["split"] = split

    meta = {
        "n_channels":         all_stats["train"]["n_channels"],
        "channel_names":      channel_names,
        "patch_size":         patch_size,
        "stride":             stride,
        "negative_margin_m":  negative_margin_m,
        "events":             [s.get("event_name") for s in scenes],
        "val_events":         list(val_events or DEFAULT_VAL_EVENTS),
        "test_events":        list(test_events or DEFAULT_TEST_EVENTS),
        "train_events":       list(train_events or []),
        "positive_fraction":  positive_fraction,
        "per_event":          per_event,
        "splits":             all_stats,
    }
    with open(output_dir / "dataset_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # The per-event scratch directory is large and fully represented by the
    # pooled splits; keeping it would double the disk footprint.
    scratch = output_dir / "_per_event"
    if scratch.exists():
        shutil.rmtree(scratch)

    log.info(
        "Event dataset written to %s  (train=%d val=%d test=%d)",
        output_dir,
        all_stats["train"]["n_patches"],
        all_stats["val"]["n_patches"],
        all_stats["test"]["n_patches"],
    )
    return all_stats


def _materialise_event_mask(
    scene_config: dict,
    work_dir: Path,
    negative_margin_m: float,
    polygon_buffer_m: float,
    patch_size: int = 256,
) -> dict:
    """
    Return a scene config guaranteed to carry a rasterised ``mask``.

    If the config already points at a mask GeoTIFF it is passed through.  If it
    points at ``polygon_file``, the inventory polygons are rasterised onto the
    Sentinel-1 grid and written to *work_dir* so the rest of the pipeline can
    treat polygons and masks identically.

    The Sentinel-1 post-event image defines the grid: it is the highest
    resolution all-weather input, so Landsat and DEM resample onto it rather
    than the other way round.

    Parameters
    ----------
    scene_config : dict
        Per-event config; needs ``mask`` or ``polygon_file`` plus ``sar_stack``,
        ``sar_post``, or an explicit ``reference`` raster.
    work_dir : Path
        Scratch directory for the intermediate mask GeoTIFF.  Removed by the
        caller once all events are processed.
    negative_margin_m, polygon_buffer_m : float
    patch_size : int
        Patch size in pixels, used as the lower bound on the negative margin.

    Returns
    -------
    dict
        A copy of *scene_config* with ``mask`` set to a raster path.
    """
    cfg = dict(scene_config)
    if cfg.get("mask"):
        return cfg

    event = cfg.get("event_name", "unknown")
    polygon_file = cfg.get("polygon_file")
    if polygon_file is None:
        raise ValueError(
            f"Event '{event}' has neither 'mask' nor 'polygon_file'. Supply a "
            f"landslide mask GeoTIFF or an inventory polygon file."
        )

    polygons, polygon_crs = load_event_polygons(polygon_file)
    log.info("Loaded %d inventory polygon(s) from %s", len(polygons), polygon_file)
    if polygon_crs:
        log.info("Inventory CRS: %s", polygon_crs[:60])

    reference = (
        cfg.get("sar_stack")
        or cfg.get("sar_post")
        or cfg.get("reference")
    )
    if reference is None:
        raise ValueError(
            f"Event '{event}' needs 'reference', 'sar_stack', or 'sar_post' to "
            f"define the grid the polygons are rasterised onto."
        )

    with rasterio.open(reference) as src:
        ref_profile = src.profile.copy()

    mask, inside_aoi = rasterize_landslide_polygons(
        polygons, ref_profile,
        buffer_m=polygon_buffer_m,
        source_crs=polygon_crs,
    )
    mask = add_negative_margin(
        mask, inside_aoi, ref_profile, negative_margin_m, min_px=patch_size
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    mask_path = work_dir / "landslide_mask.tif"
    mask_profile = ref_profile.copy()
    # Drop the reference's tiling: a small mask inherited with the source's
    # block shape can exceed the raster size or contradict its own TILED flag,
    # and GDAL warns or errors rather than writing the file.
    mask_profile.pop("blockxsize", None)
    mask_profile.pop("blockysize", None)
    mask_profile.update(
        dtype="int8", count=1, nodata=-1, compress="deflate", tiled=False,
    )
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask.astype(np.int8), 1)
    log.info(
        "Wrote mask raster: %s  (%d positives, %d valid negatives)",
        mask_path,
        int((mask == 1).sum()),
        int((mask == 0).sum()),
    )

    cfg["mask"] = str(mask_path)
    return cfg


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Landslide dataset preparation — Sentinel-1 + Landsat event catalog "
            "(or the legacy Landslide4Sense benchmark)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-event S1 + Landsat dataset (active pipeline)
  python prepare_dataset.py events --config data/config/events.json \\
      --output data/processed/s1_landsat

  # Single custom India scene
  python prepare_dataset.py custom --config scene_config.json --output data/processed/wayanad

  # Legacy Landslide4Sense benchmark (Sentinel-2 based)
  python prepare_dataset.py l4s --l4s-dir data/l4s --output data/processed/l4s

Example events.json (one entry per landslide event):
  {
    "events": [
      {
        "event_name": "wayanad_2024",
        "lat": 11.51, "lon": 76.15,
        "polygon_file": "data/inventory/wayanad_2024.geojson",
        "sar_stack": "output/sar/wayanad_2024/change/sar_stack.tif",
        "landsat":   "output/optical/wayanad_2024/post/stack.tif",
        "dem":       "output/dem/wayanad_2024/terrain_stack.tif"
      }
    ],
    "val_events":  ["tupul_2022", "pettimudi_2020"],
    "test_events": ["irshalwadi_2023", "joshimath_2023"]
  }

Example scene_config.json (single scene):
  {
    "event_name": "wayanad_2024",
    "lat": 11.51, "lon": 76.15,
    "sar_stack": "output/sar/wayanad/change/sar_stack.tif",
    "landsat":   "output/optical/wayanad/post/stack.tif",
    "dem":       "output/dem/wayanad/terrain_stack.tif",
    "mask":      "data/inventory/wayanad_mask.tif"
  }
""",
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # --- events sub-command (active pipeline) ---
    ev_p = sub.add_parser(
        "events",
        help="Build a multi-event Sentinel-1 + Landsat patch dataset",
    )
    ev_p.add_argument(
        "--config", required=True,
        help="JSON file with an 'events' list (plus optional val/test event keys).",
    )
    ev_p.add_argument("--output", required=True, help="Output dataset root")
    ev_p.add_argument(
        "--patch-size", type=int, default=256,
        help="Patch size in pixels (default 256).",
    )
    ev_p.add_argument(
        "--stride", type=int, default=128,
        help="Sliding-window stride in pixels (default 128).",
    )
    ev_p.add_argument(
        "--positive-fraction", type=float, default=0.6,
        help="Target fraction of positive patches (default 0.6).",
    )
    ev_p.add_argument(
        "--negative-margin", type=float, default=150.0, metavar="M",
        help=(
            "Width in metres of the valid-negative ring around each inventory "
            "polygon (default 150)."
        ),
    )
    ev_p.add_argument(
        "--polygon-buffer", type=float, default=0.0, metavar="M",
        help="Buffer polygons before rasterising, in metres (default 0).",
    )
    ev_p.add_argument(
        "--val-events", nargs="+", default=None, metavar="EVENT",
        help=(
            "Events held out for validation. Overrides the config's "
            "'val_events'; defaults to "
            + ", ".join(DEFAULT_VAL_EVENTS) + "."
        ),
    )
    ev_p.add_argument(
        "--test-events", nargs="+", default=None, metavar="EVENT",
        help=(
            "Events held out for testing. Overrides the config's "
            "'test_events'; defaults to "
            + ", ".join(DEFAULT_TEST_EVENTS) + "."
        ),
    )
    ev_p.add_argument(
        "--train-events", nargs="*", default=None, metavar="EVENT",
        help=(
            "Force these events into train (everything not held out is train "
            "anyway; supplying this only asserts the split is intentional)."
        ),
    )

    # --- l4s sub-command (legacy) ---
    l4s_p = sub.add_parser(
        "l4s",
        help="Process Landslide4Sense benchmark dataset (legacy, Sentinel-2 based)",
    )
    l4s_p.add_argument("--l4s-dir",    required=True, help="Root directory of L4S dataset")
    l4s_p.add_argument("--output",     required=True, help="Output directory for processed patches")
    l4s_p.add_argument("--patch-size", type=int, default=128, help="Expected patch size (default 128)")

    # --- custom sub-command ---
    cust_p = sub.add_parser("custom", help="Process a single custom India scene")
    cust_p.add_argument("--config",     required=True, help="Path to scene_config.json")
    cust_p.add_argument("--output",     required=True, help="Output directory for processed patches")
    cust_p.add_argument("--patch-size", type=int, default=256, help="Patch size (default 256)")
    cust_p.add_argument("--stride",     type=int, default=128, help="Sliding window stride (default 128)")

    # Global
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mode == "l4s":
        stats = prepare_l4s_dataset(
            l4s_dir=args.l4s_dir,
            output_dir=args.output,
            patch_size=args.patch_size,
        )
        print("\n✓ L4S dataset prepared:")
        for split, s in stats.items():
            print(
                f"  {split:5s}: {s['n_patches']:4d} patches  "
                f"(pos={s['n_positive']}  neg={s['n_negative']})"
            )

    elif args.mode == "events":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)

        events = config["events"] if isinstance(config, dict) else config
        # CLI flags win over the config file, which wins over the defaults.
        val_events = (
            args.val_events
            or (config.get("val_events") if isinstance(config, dict) else None)
        )
        test_events = (
            args.test_events
            or (config.get("test_events") if isinstance(config, dict) else None)
        )
        train_events = (
            args.train_events
            or (config.get("train_events") if isinstance(config, dict) else None)
        )

        stats = prepare_event_dataset(
            scenes=events,
            output_dir=args.output,
            patch_size=args.patch_size,
            stride=args.stride,
            positive_fraction=args.positive_fraction,
            val_events=val_events,
            test_events=test_events,
            train_events=train_events,
            negative_margin_m=args.negative_margin,
            polygon_buffer_m=args.polygon_buffer,
        )
        print("\n✓ S1 + Landsat event dataset prepared:")
        for split, s in stats.items():
            print(
                f"  {split:5s}: {s['n_patches']:4d} patches  "
                f"(pos={s['n_positive']}  neg={s['n_negative']})  "
                f"channels={s['n_channels']}"
            )

    elif args.mode == "custom":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path) as f:
            scene_config = json.load(f)

        stats = prepare_custom_india_dataset(
            scene_config=scene_config,
            output_dir=args.output,
            patch_size=args.patch_size,
            stride=args.stride,
        )
        print("\n✓ Custom India dataset prepared:")
        for split, s in stats.items():
            print(
                f"  {split:5s}: {s['n_patches']:4d} patches  "
                f"(pos={s['n_positive']}  neg={s['n_negative']})"
            )


if __name__ == "__main__":
    main()
