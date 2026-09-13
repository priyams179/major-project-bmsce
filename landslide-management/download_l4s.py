#!/usr/bin/env python3
"""
Landslide4Sense (L4S) Dataset Downloader

Downloads the Landslide4Sense benchmark dataset:
  - 3,799 multi-spectral image patches (14 channels: 12 Sentinel-2 + elevation + slope)
  - Binary pixel-level landslide masks
  - 4 global regions: Japan, Taiwan, India, Nepal
  - Pre-split into train/val/test sets

Dataset: IEEE DataPort / Zenodo
Paper: Ghorbanzadeh et al., IEEE TGRS 2022
Note: Dataset requires free registration at IEEE DataPort.
      This script provides instructions and validates downloaded files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# HDF5 / rasterio are optional for validation
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("download_l4s")

# ===========================================================================
# Dataset information
# ===========================================================================

DATASET_INFO: dict = {
    "name": "Landslide4Sense",
    "version": "1.0",
    "paper": "Ghorbanzadeh et al., IEEE Transactions on Geoscience and Remote Sensing, 2022",
    "doi":   "10.1109/TGRS.2022.3215209",

    # Patch dimensions
    "patch_size":   128,
    "n_channels":   14,
    "mask_classes": {0: "Background (no landslide)", 1: "Landslide", -1: "Invalid/No-data"},

    # Channel descriptions (0-indexed)
    "channels": {
        0:  "Sentinel-2 B01 – Coastal aerosol (60 m native, resampled to 10 m)",
        1:  "Sentinel-2 B02 – Blue          (10 m)",
        2:  "Sentinel-2 B03 – Green         (10 m)",
        3:  "Sentinel-2 B04 – Red           (10 m)",
        4:  "Sentinel-2 B05 – Red Edge 1    (20 m, resampled)",
        5:  "Sentinel-2 B06 – Red Edge 2    (20 m, resampled)",
        6:  "Sentinel-2 B07 – Red Edge 3    (20 m, resampled)",
        7:  "Sentinel-2 B08 – NIR broad     (10 m)",
        8:  "Sentinel-2 B8A – NIR narrow    (20 m, resampled)",
        9:  "Sentinel-2 B09 – Water vapour  (60 m, resampled)",
        10: "Sentinel-2 B11 – SWIR 1        (20 m, resampled)",
        11: "Sentinel-2 B12 – SWIR 2        (20 m, resampled)",
        12: "ALOS PALSAR DEM – Elevation    (m, 12.5 m native resampled to 10 m)",
        13: "Slope (degrees, derived from ALOS PALSAR DEM)",
    },

    # Expected patch counts per split (L4S v1.0)
    "split_counts": {
        "train": 3799,   # full dataset; L4S paper uses 80/10/10 split internally
        "val":    245,
        "test":   245,
    },

    # Regions represented in the dataset
    "regions": ["Japan", "Taiwan", "India", "Nepal"],

    # Download sources
    "urls": {
        "ieee_dataport": "https://ieee-dataport.org/open-access/landslide4sense-reference-benchmark-data-set-geospatial-ai",
        "zenodo":        "https://zenodo.org/record/7487743",
        "huggingface":   "https://huggingface.co/datasets/satellogic/landslide4sense",
        "gee":           "https://developers.google.com/earth-engine/datasets/catalog/landslide4sense",
    },
}


# ===========================================================================
# 1. Structural validation
# ===========================================================================

def check_l4s_structure(data_dir: str | Path) -> dict:
    """
    Verify that the L4S dataset is correctly downloaded and structured.

    Checks both Layout A (split subdirs) and Layout B (splits.json) arrangements.

    Parameters
    ----------
    data_dir : Root of L4S dataset.

    Returns
    -------
    dict:
      {
        'valid'       : bool,
        'layout'      : str   'A' | 'B' | 'unknown',
        'train_count' : int,
        'val_count'   : int,
        'test_count'  : int,
        'issues'      : list[str],
      }
    """
    data_dir = Path(data_dir)
    issues: List[str] = []
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    layout = "unknown"

    if not data_dir.exists():
        return {
            "valid": False,
            "layout": layout,
            "train_count": 0,
            "val_count":   0,
            "test_count":  0,
            "issues": [f"Directory does not exist: {data_dir}"],
        }

    # --- Layout A check -----------------------------------------------------
    layout_a_ok = True
    for split in ("train", "val", "test"):
        img_dir  = data_dir / split / "image"
        mask_dir = data_dir / split / "mask"
        if not img_dir.is_dir():
            layout_a_ok = False
            break
        img_files  = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.h5"))
        mask_files = list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.h5"))
        counts[split] = len(img_files)

        if len(img_files) != len(mask_files):
            issues.append(
                f"[{split}] Image count ({len(img_files)}) ≠ Mask count ({len(mask_files)})"
            )

        expected = DATASET_INFO["split_counts"].get(split, 0)
        if counts[split] < expected:
            issues.append(
                f"[{split}] Found {counts[split]} patches, expected ≥ {expected}."
            )

    if layout_a_ok:
        layout = "A"

    # --- Layout B check (splits.json) ----------------------------------------
    splits_json = data_dir / "splits.json"
    if layout == "unknown" and splits_json.exists():
        layout = "B"
        try:
            with open(splits_json) as f:
                sdata = json.load(f)
            for split in ("train", "val", "test"):
                counts[split] = len(sdata.get(split, []))
        except (json.JSONDecodeError, KeyError) as e:
            issues.append(f"splits.json parse error: {e}")

    if layout == "unknown":
        # Flat structure?
        img_dir = data_dir / "image"
        if img_dir.is_dir():
            all_imgs = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.h5"))
            counts["train"] = len(all_imgs)
            issues.append(
                "Dataset appears flat (no train/val/test subdirs and no splits.json). "
                "Run --organize to restructure."
            )
            layout = "flat"
        else:
            issues.append(
                "Cannot detect dataset structure. "
                "Expected: train/image/*.tif OR splits.json OR image/*.tif"
            )

    total = sum(counts.values())
    if total == 0:
        issues.append("No patch files found in any split.")

    is_valid = len(issues) == 0 and total > 0

    result = {
        "valid":       is_valid,
        "layout":      layout,
        "train_count": counts["train"],
        "val_count":   counts["val"],
        "test_count":  counts["test"],
        "total":       total,
        "issues":      issues,
    }

    if is_valid:
        log.info("L4S structure OK: %s (layout=%s)", data_dir, layout)
    else:
        log.warning("L4S structure issues found:")
        for iss in issues:
            log.warning("  • %s", iss)

    return result


# ===========================================================================
# 2. Download instructions
# ===========================================================================

def print_download_instructions() -> None:
    """Print step-by-step instructions for obtaining the L4S dataset."""

    banner = "=" * 72
    print(f"\n{banner}")
    print("  Landslide4Sense (L4S) — Dataset Download Instructions")
    print(f"{banner}\n")

    print("OPTION 1 — Zenodo (Recommended, no login required for basic download)")
    print("-" * 60)
    print("  URL  : https://zenodo.org/record/7487743")
    print("  Steps:")
    print("    1. Open the URL in your browser.")
    print("    2. Scroll to 'Files' and download:")
    print("       • TrainData.zip  (~900 MB)")
    print("       • ValidData.zip")
    print("       • TestData.zip")
    print("    3. Extract all archives:")
    print("       unzip TrainData.zip -d data/l4s/")
    print("       unzip ValidData.zip -d data/l4s/")
    print("       unzip TestData.zip  -d data/l4s/")
    print("    4. Run this script to organize:")
    print("       python download_l4s.py --organize --raw-dir data/l4s/raw --data-dir data/l4s\n")

    print("OPTION 2 — IEEE DataPort")
    print("-" * 60)
    print("  URL  : https://ieee-dataport.org/open-access/landslide4sense-reference-"
          "benchmark-data-set-geospatial-ai")
    print("  Note : Requires FREE IEEE account registration.")
    print("  Steps:")
    print("    1. Register/login at https://ieee-dataport.org")
    print("    2. Navigate to the dataset page and accept the license.")
    print("    3. Download all .zip files listed.")
    print("    4. Extract and run --organize as above.\n")

    print("OPTION 3 — Hugging Face Hub (Python API, easiest)")
    print("-" * 60)
    print("  pip install huggingface_hub")
    print("  python - <<'EOF'")
    print("  from huggingface_hub import snapshot_download")
    print("  snapshot_download(")
    print('      repo_id="satellogic/landslide4sense",')
    print('      repo_type="dataset",')
    print('      local_dir="data/l4s"')
    print("  )")
    print("  EOF\n")

    print("OPTION 4 — Google Earth Engine (GEE)")
    print("-" * 60)
    print("  The dataset is available as a GEE asset for Earth Engine users.")
    print("  See: https://developers.google.com/earth-engine/datasets/catalog/landslide4sense")
    print("  Use the GEE Python API (earthengine-api) to export patches.\n")

    print("OPTION 5 — wget (Zenodo direct links)")
    print("-" * 60)
    print("  # Replace <record-id> with the actual Zenodo record ID")
    print("  BASE=https://zenodo.org/record/7487743/files")
    print("  wget $BASE/TrainData.zip -P data/l4s/raw/")
    print("  wget $BASE/ValidData.zip -P data/l4s/raw/")
    print("  wget $BASE/TestData.zip  -P data/l4s/raw/\n")

    print("After downloading, validate the structure with:")
    print("  python download_l4s.py --check --data-dir data/l4s\n")
    print(banner)


# ===========================================================================
# 3. Directory organizer
# ===========================================================================

def organize_l4s_directory(
    raw_dir: str | Path,
    output_dir: str | Path,
) -> bool:
    """
    Organize a downloaded L4S directory into the expected layout::

        output_dir/
          train/image/*.tif   train/mask/*.tif
          val/image/*.tif     val/mask/*.tif
          test/image/*.tif    test/mask/*.tif

    Handles:
      - Flat directories (all images together)
      - Separate train/val/test folders but without image/mask sub-folders
      - Mixed HDF5 (.h5) and GeoTIFF (.tif) files

    Parameters
    ----------
    raw_dir    : Where the raw downloaded files are.
    output_dir : Where to write the organised structure.

    Returns
    -------
    bool  True if organisation succeeded without errors.
    """
    import shutil

    raw_dir    = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []

    def _collect_pairs(search_root: Path) -> List[Tuple[Path, Path]]:
        """Return (image_path, mask_path) pairs from a directory."""
        # Strategy: assume images are in an 'image' sub-dir and masks in 'mask',
        # OR that mask files have 'mask' in their name and image files do not.
        img_dir  = search_root / "image"
        mask_dir = search_root / "mask"

        if img_dir.is_dir() and mask_dir.is_dir():
            imgs  = sorted(img_dir.glob("*.tif")) + sorted(img_dir.glob("*.h5"))
            masks = sorted(mask_dir.glob("*.tif")) + sorted(mask_dir.glob("*.h5"))
            return list(zip(imgs, masks))

        # Try name-based pairing
        all_files = sorted(
            list(search_root.glob("*.tif"))
            + list(search_root.glob("*.h5"))
        )
        imgs  = [f for f in all_files if "mask" not in f.stem.lower()]
        masks = [f for f in all_files if "mask" in f.stem.lower()]

        if len(imgs) == len(masks):
            return list(zip(imgs, masks))

        # Last resort: assume alternating (img0, mask0, img1, mask1 …)
        if len(all_files) % 2 == 0:
            imgs  = all_files[0::2]
            masks = all_files[1::2]
            return list(zip(imgs, masks))

        return []

    # Detect splits
    split_map: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        candidate = raw_dir / split
        if candidate.is_dir():
            split_map[split] = candidate

    if not split_map:
        # Flat: try to infer from filenames or put everything in train
        log.warning(
            "No train/val/test sub-directories found in %s. "
            "Treating all files as 'train'.", raw_dir,
        )
        split_map["train"] = raw_dir

    success = True
    for split, src_root in split_map.items():
        pairs = _collect_pairs(src_root)
        if not pairs:
            log.warning("No image/mask pairs found under %s", src_root)
            continue

        out_img_dir  = output_dir / split / "image"
        out_mask_dir = output_dir / split / "mask"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_mask_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_src, mask_src) in enumerate(pairs):
            stem = f"{idx:05d}"
            ext  = img_src.suffix

            img_dst  = out_img_dir  / f"{stem}{ext}"
            mask_dst = out_mask_dir / f"{stem}{ext}"

            try:
                shutil.copy2(img_src,  img_dst)
                shutil.copy2(mask_src, mask_dst)
            except Exception as exc:
                errors.append(str(exc))
                success = False

        log.info(
            "Organized %s: %d pairs → %s", split, len(pairs), output_dir / split
        )

    if errors:
        log.error("Errors during organization:")
        for e in errors:
            log.error("  %s", e)
        success = False

    return success


# ===========================================================================
# 4. Patch validator
# ===========================================================================

def validate_l4s_patch(
    image_path: str | Path,
    mask_path: Optional[str | Path] = None,
) -> dict:
    """
    Validate a single L4S patch for correctness.

    Checks:
      • Shape matches (128, 128, 14)
      • Reflectance channels are in a plausible range (not all zero or NaN)
      • Elevation (ch 12) has realistic values for South/East Asia
      • Slope (ch 13) is non-negative
      • Mask values are only in {-1, 0, 1}

    Parameters
    ----------
    image_path : Path to image patch
    mask_path  : Path to companion mask (optional; inferred if None)

    Returns
    -------
    dict:
      {
        'valid'   : bool,
        'shape'   : tuple,
        'n_channels': int,
        'issues'  : list[str],
        'stats'   : dict   per-channel min/max/mean,
      }
    """
    image_path = Path(image_path)
    issues: List[str] = []

    # Load image
    if image_path.suffix.lower() in (".h5", ".hdf5"):
        if not HAS_H5PY:
            return {"valid": False, "issues": ["h5py not installed"]}
        with h5py.File(image_path) as f:
            keys = list(f.keys())
            img_key = "img" if "img" in keys else (keys[0] if keys else None)
            if img_key is None:
                return {"valid": False, "issues": ["Empty HDF5 file"]}
            image = np.array(f[img_key], dtype=np.float32)
    else:
        if not HAS_RASTERIO:
            return {"valid": False, "issues": ["rasterio not installed"]}
        with rasterio.open(image_path) as src:
            data = src.read().astype(np.float32)  # (C, H, W)
        # If last band could be mask, check channel count
        if data.shape[0] == 15:
            image = np.moveaxis(data[:14], 0, -1)
        elif data.shape[0] == 14:
            image = np.moveaxis(data, 0, -1)
        else:
            image = np.moveaxis(data, 0, -1)

    # Validate shape
    expected_shape = (128, 128, 14)
    if image.shape != expected_shape:
        issues.append(
            f"Shape mismatch: got {image.shape}, expected {expected_shape}"
        )

    # Per-channel statistics
    stats: Dict[str, dict] = {}
    if image.ndim == 3:
        n_ch = image.shape[2]
        for ch in range(min(n_ch, 14)):
            ch_data = image[:, :, ch]
            has_nan = bool(np.isnan(ch_data).any())
            stats[f"ch{ch:02d}"] = {
                "min": float(np.nanmin(ch_data)),
                "max": float(np.nanmax(ch_data)),
                "mean": float(np.nanmean(ch_data)),
                "has_nan": has_nan,
            }
            if has_nan:
                issues.append(f"Channel {ch} contains NaN values.")
            if ch < 12:  # Reflectance channels
                ch_min = float(np.nanmin(ch_data))
                ch_max = float(np.nanmax(ch_data))
                if ch_max > 20_000 or ch_min < -5_000:
                    issues.append(
                        f"Channel {ch} (reflectance) has extreme values: "
                        f"[{ch_min:.1f}, {ch_max:.1f}] (expected near [0, 10000])."
                    )
            elif ch == 13:  # Slope
                if float(np.nanmin(ch_data)) < -0.01:
                    issues.append(f"Channel {ch} (slope) has negative values.")

    # Validate mask
    mask_data: Optional[np.ndarray] = None
    if mask_path is None:
        # Try to infer mask path
        candidate = Path(str(image_path).replace("/image/", "/mask/"))
        if candidate.exists():
            mask_path = candidate

    if mask_path is not None:
        mask_path = Path(mask_path)
        if mask_path.suffix.lower() in (".h5", ".hdf5"):
            if HAS_H5PY:
                with h5py.File(mask_path) as f:
                    mk = list(f.keys())
                    mask_data = np.array(f[mk[0]], dtype=np.int8)
        else:
            if HAS_RASTERIO:
                with rasterio.open(mask_path) as src:
                    mask_data = src.read(1).astype(np.int8)

    if mask_data is not None:
        unique_vals = set(np.unique(mask_data).tolist())
        invalid_vals = unique_vals - {-1, 0, 1}
        if invalid_vals:
            issues.append(
                f"Mask contains unexpected values: {invalid_vals}. "
                "Expected only {{-1, 0, 1}}."
            )
        n_positive = int((mask_data == 1).sum())
        n_invalid  = int((mask_data == -1).sum())
        stats["mask"] = {
            "n_positive": n_positive,
            "n_negative": int((mask_data == 0).sum()),
            "n_invalid":  n_invalid,
        }

    is_valid = len(issues) == 0
    return {
        "valid":      is_valid,
        "shape":      tuple(image.shape),
        "n_channels": image.shape[2] if image.ndim == 3 else 0,
        "issues":     issues,
        "stats":      stats,
    }


# ===========================================================================
# 5. Dataset statistics
# ===========================================================================

def generate_dataset_statistics(data_dir: str | Path) -> dict:
    """
    Compute patch counts and class balance per split.

    Parameters
    ----------
    data_dir : Root of the L4S dataset.

    Returns
    -------
    dict with per-split counts and class statistics (sampled if large).
    """
    data_dir = Path(data_dir)
    structure = check_l4s_structure(data_dir)

    stats: dict = {
        "data_dir": str(data_dir),
        "structure": structure,
        "splits":   {},
    }

    SAMPLE_SIZE = 50  # Max patches to inspect per split for class balance

    for split in ("train", "val", "test"):
        count = structure.get(f"{split}_count", 0)
        if count == 0:
            stats["splits"][split] = {"n_patches": 0}
            continue

        # Locate patches
        if structure["layout"] == "A":
            img_dir  = data_dir / split / "image"
            mask_dir = data_dir / split / "mask"
            imgs  = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.h5")))
            masks = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.h5")))
        elif structure["layout"] in ("B", "flat"):
            img_dir  = data_dir / "image"
            mask_dir = data_dir / "mask"
            imgs  = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.h5")))
            masks = sorted(list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.h5")))
        else:
            stats["splits"][split] = {"n_patches": count, "note": "layout unknown"}
            continue

        sample_n = min(SAMPLE_SIZE, len(imgs))
        n_positive = 0
        n_total_ls_px = 0
        n_total_px    = 0

        for img_p, mask_p in zip(imgs[:sample_n], masks[:sample_n]):
            report = validate_l4s_patch(img_p, mask_p)
            mstats = report.get("stats", {}).get("mask", {})
            n_positive    += 1 if mstats.get("n_positive", 0) > 0 else 0
            n_total_ls_px += mstats.get("n_positive", 0)
            n_total_px    += 128 * 128

        stats["splits"][split] = {
            "n_patches":                count,
            "sampled":                  sample_n,
            "positive_patch_fraction":  round(n_positive / max(sample_n, 1), 3),
            "landslide_pixel_fraction": round(n_total_ls_px / max(n_total_px, 1), 5),
        }

    # Print summary table
    print("\n" + "=" * 62)
    print(f"  L4S Dataset Statistics — {data_dir}")
    print("=" * 62)
    print(f"  Structure layout : {structure['layout']}")
    print(f"  Issues           : {structure['issues'] or 'None'}")
    print()
    print(f"  {'Split':<8} {'Patches':>8} {'Pos.Patch%':>12} {'Slide px%':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for split in ("train", "val", "test"):
        s = stats["splits"].get(split, {})
        n_p = s.get("n_patches", 0)
        pp  = s.get("positive_patch_fraction", float("nan"))
        lp  = s.get("landslide_pixel_fraction", float("nan"))
        print(f"  {split:<8} {n_p:>8} {pp:>11.1%} {lp:>11.3%}")
    print("=" * 62 + "\n")

    return stats


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Landslide4Sense dataset checker, organizer, and statistics generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if dataset is correctly downloaded
  python download_l4s.py --check --data-dir data/l4s

  # Print download instructions
  python download_l4s.py --instructions

  # Organize raw downloaded files into expected structure
  python download_l4s.py --organize --raw-dir ~/Downloads/L4S --data-dir data/l4s

  # Generate dataset statistics and class balance report
  python download_l4s.py --stats --data-dir data/l4s
""",
    )

    p.add_argument("--check",        action="store_true",
                   help="Check L4S dataset structure and report issues")
    p.add_argument("--instructions", action="store_true",
                   help="Print step-by-step download instructions")
    p.add_argument("--organize",     action="store_true",
                   help="Organize raw downloaded files into expected directory layout")
    p.add_argument("--stats",        action="store_true",
                   help="Generate and print dataset statistics")
    p.add_argument("--validate-patch", metavar="IMAGE_PATH",
                   help="Validate a single patch file")

    p.add_argument("--data-dir",  default="data/l4s",
                   help="Root directory of L4S dataset (default: data/l4s)")
    p.add_argument("--raw-dir",   default=None,
                   help="Directory containing raw downloaded files (for --organize)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not any([args.check, args.instructions, args.organize,
                args.stats, args.validate_patch]):
        # Default: show instructions
        print_download_instructions()
        return

    if args.instructions:
        print_download_instructions()

    if args.check:
        result = check_l4s_structure(args.data_dir)
        print(f"\nL4S Structure Check — {args.data_dir}")
        print(f"  Valid  : {result['valid']}")
        print(f"  Layout : {result['layout']}")
        print(f"  Train  : {result['train_count']} patches")
        print(f"  Val    : {result['val_count']} patches")
        print(f"  Test   : {result['test_count']} patches")
        if result["issues"]:
            print("\n  Issues:")
            for iss in result["issues"]:
                print(f"    • {iss}")
        else:
            print("  ✓ No issues found.")

    if args.organize:
        if args.raw_dir is None:
            print(
                "[ERROR] --raw-dir is required for --organize. "
                "Specify the directory containing the raw downloaded files.",
                file=sys.stderr,
            )
            sys.exit(1)
        ok = organize_l4s_directory(args.raw_dir, args.data_dir)
        if ok:
            print(f"\n✓ Organization successful. Dataset structured at: {args.data_dir}")
        else:
            print("\n✗ Organization completed with errors. Check logs above.", file=sys.stderr)

    if args.stats:
        generate_dataset_statistics(args.data_dir)

    if args.validate_patch:
        report = validate_l4s_patch(args.validate_patch)
        print(f"\nPatch Validation — {args.validate_patch}")
        print(f"  Valid     : {report['valid']}")
        print(f"  Shape     : {report['shape']}")
        print(f"  Channels  : {report['n_channels']}")
        if report["issues"]:
            print("  Issues:")
            for iss in report["issues"]:
                print(f"    • {iss}")
        else:
            print("  ✓ Patch looks healthy.")
        if report.get("stats"):
            mask_s = report["stats"].get("mask")
            if mask_s:
                print(
                    f"  Mask      : {mask_s['n_positive']} landslide px  "
                    f"/ {mask_s['n_negative']} background  "
                    f"/ {mask_s['n_invalid']} invalid"
                )


if __name__ == "__main__":
    main()
