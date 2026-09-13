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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
L4S_PATCH_SIZE = 128
L4S_CHANNELS   = 14

# Minimum fraction of a patch that must be covered by valid (non-cloud) pixels
MIN_VALID_COVERAGE = 0.10

# ===========================================================================
# 1. Landslide4Sense patch loading
# ===========================================================================

def load_l4s_patch(h5_or_tif_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single Landslide4Sense (L4S) patch.

    The L4S dataset provides 128×128 patches with 14 channels:
      ch 0-11  – Sentinel-2 BOA reflectance bands B01…B12 (excluding B10)
      ch 12    – ALOS PALSAR DEM elevation (m)
      ch 13    – Slope (degrees)

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

def stack_multimodal_bands(
    optical_bands: Optional[np.ndarray] = None,
    sar_bands: Optional[np.ndarray] = None,
    dem_bands: Optional[np.ndarray] = None,
    indices_bands: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Concatenate arrays from different modalities along the channel axis.

    All input arrays must have the same (H, W) spatial footprint.
    Arrays can be:
      - 2-D (H, W)   → treated as 1 channel
      - 3-D (H, W, C) → multi-channel

    Parameters
    ----------
    optical_bands  : Optical reflectance channels, e.g. (H, W, 9)
    sar_bands      : SAR channels (VV, VH, ratio, coherence …), e.g. (H, W, 6)
    dem_bands      : Elevation / slope / aspect …, e.g. (H, W, 2)
    indices_bands  : Pre-computed indices (NDVI, dNDVI, …), e.g. (H, W, 3)

    Returns
    -------
    np.ndarray (H, W, N_total_channels), float32.
    """
    layers: List[np.ndarray] = []
    names:  List[str]        = []

    for arr, name in [
        (optical_bands, "optical"),
        (sar_bands,     "sar"),
        (dem_bands,     "dem"),
        (indices_bands, "indices"),
    ]:
        if arr is None:
            continue
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim != 3:
            raise ValueError(
                f"Band array for '{name}' must be 2-D or 3-D, got {arr.ndim}-D."
            )
        layers.append(arr)
        names.append(f"{name}({arr.shape[2]}ch)")

    if not layers:
        raise ValueError("At least one modality must be provided.")

    # Validate spatial dimensions match
    shapes = [a.shape[:2] for a in layers]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"Spatial shapes do not match across modalities: "
            + ", ".join(f"{n}: {s}" for n, s in zip(names, shapes))
        )

    stack = np.concatenate(layers, axis=-1)
    log.info("Multi-modal stack: %s → shape=%s", " + ".join(names), stack.shape)
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

def spatial_block_split(
    patches_with_coords: List[dict],
    val_regions: Optional[List[str]] = None,
    test_regions: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split patches by geographic region to prevent spatial data leakage.

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
        "Spatial split: train=%d  val=%d  test=%d",
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
    patches    : list of patch dicts
    output_dir : Path
    prefix     : str  prepended to every filename (e.g. 'train_')

    Returns
    -------
    dict  statistics:
      {
        'n_patches'        : int,
        'n_positive'       : int,
        'n_negative'       : int,
        'positive_fraction': float,
        'output_dir'       : str,
      }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        })

    splits_path = output_dir / f"{prefix}splits.json"
    with open(splits_path, "w") as f:
        json.dump({"patches": meta_records}, f, indent=2)

    n_total = len(patches)
    stats = {
        "n_patches":         n_total,
        "n_positive":        n_positive,
        "n_negative":        n_total - n_positive,
        "positive_fraction": n_positive / max(n_total, 1),
        "output_dir":        str(output_dir),
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
        'mask'     : Path to binary landslide mask GeoTIFF (0/1)
      Optional (at least one modality required):
        'sar_pre'  : Path to pre-event SAR GeoTIFF  (1 or more bands)
        'sar_post' : Path to post-event SAR GeoTIFF
        'optical'  : Path to optical stack GeoTIFF  (from preprocess_optical)
        'dem'      : Path to DEM/slope GeoTIFF
      Metadata (optional):
        'event_name' : str
        'lat'        : float   scene centre latitude
        'lon'        : float   scene centre longitude
        'region'     : str     override region assignment

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
        ("sar_pre",  "bilinear"),
        ("sar_post", "bilinear"),
        ("optical",  "bilinear"),
        ("dem",      "bilinear"),
    ]:
        arr = _load_and_align(scene_config.get(key), key, rs)
        if arr is not None:
            modality_arrays[key] = arr

    if not modality_arrays:
        raise ValueError(
            "scene_config must include at least one of: sar_pre, sar_post, optical, dem"
        )

    # ------------------------------------------------------------------
    # Stack all available modalities
    # ------------------------------------------------------------------
    full_stack = stack_multimodal_bands(
        optical_bands = modality_arrays.get("optical"),
        sar_bands     = _hstack_if_both(
                            modality_arrays.get("sar_pre"),
                            modality_arrays.get("sar_post"),
                        ),
        dem_bands     = modality_arrays.get("dem"),
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
        p["region"]     = region
        p["event_name"] = event_name
        p["lat"]        = scene_config.get("lat")
        p["lon"]        = scene_config.get("lon")

    # Balanced sampling
    balanced = balanced_patch_sampling(all_patches, positive_fraction=0.6)

    # Spatial block split
    train_p, val_p, test_p = spatial_block_split(balanced)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    all_stats: dict = {}
    for split, patches in [("train", train_p), ("val", val_p), ("test", test_p)]:
        stats = save_dataset(patches, output_dir / split, prefix=f"{split}_")
        stats["split"] = split
        all_stats[split] = stats

    # Write scene config metadata
    meta = {
        "event_name": event_name,
        "region":     region,
        "patch_size": patch_size,
        "stride":     stride,
        "n_channels": full_stack.shape[2],
        "modalities": list(modality_arrays.keys()),
        "splits":     all_stats,
    }
    meta_path = output_dir / "scene_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Custom India dataset done: %s", meta)
    return all_stats


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
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Landslide dataset preparation — L4S benchmark or custom India scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare Landslide4Sense benchmark
  python prepare_dataset.py l4s --l4s-dir data/l4s --output data/processed/l4s

  # Prepare custom India event from config JSON
  python prepare_dataset.py custom --config scene_config.json --output data/processed/wayanad

Example scene_config.json:
  {
    "event_name": "wayanad_2024",
    "lat": 11.51, "lon": 76.15,
    "sar_pre":  "data/raw/wayanad/sar_pre.tif",
    "sar_post": "data/raw/wayanad/sar_post.tif",
    "optical":  "data/raw/wayanad/optical_stack.tif",
    "dem":      "data/raw/wayanad/dem.tif",
    "mask":     "data/raw/wayanad/landslide_mask.tif"
  }
""",
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # --- l4s sub-command ---
    l4s_p = sub.add_parser("l4s", help="Process Landslide4Sense benchmark dataset")
    l4s_p.add_argument("--l4s-dir",    required=True, help="Root directory of L4S dataset")
    l4s_p.add_argument("--output",     required=True, help="Output directory for processed patches")
    l4s_p.add_argument("--patch-size", type=int, default=128, help="Expected patch size (default 128)")

    # --- custom sub-command ---
    cust_p = sub.add_parser("custom", help="Process a custom India scene")
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
