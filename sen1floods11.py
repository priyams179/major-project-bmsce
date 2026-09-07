#!/usr/bin/env python3
"""
Sen1Floods11 Dataset Integration & Multi-Dataset Harmonization.

Provides automated downloading and PyTorch Dataset / DataLoader classes for
the Sen1Floods11 benchmark dataset (v1.1 hand-labeled flood events and permanent water chips).

Sen1Floods11 specifications:
  - 446 hand-labeled 512x512 flood chips across 11 global disasters + Bolivia
  - 814 permanent water chips (JRC surface water + Sentinel-1 SAR)
  - Sentinel-1 SAR: Band 1 = VV, Band 2 = VH
  - Ground truth labels: 0 = not flooded, 1 = flooded, -1 = ignore (cloud / nodata)
  - Supports 2-channel [VV, VH] or 3-channel [VV, VH, (VH - VV) difference ratio]

Usage:
    uv run python sen1floods11.py --download --include-perm
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
GCS_BASE_URL = "https://storage.googleapis.com/sen1floods11/v1.1"
DEFAULT_DATA_DIR = "data/sen1floods11"

SPLIT_FILES = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "val": "splits/flood_handlabeled/flood_valid_data.csv",
    "test": "splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": "splits/flood_handlabeled/flood_bolivia_data.csv",
    "perm_train": "splits/perm_water/permanent_water_train_data.csv",
    "perm_val": "splits/perm_water/permanent_water_validation_data.csv",
    "perm_test": "splits/perm_water/permanent_water_test_data.csv",
}

# dB Normalization range for Sentinel-1
DB_MIN = -35.0
DB_MAX = 0.0


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------
def download_url(url: str, dest_path: str, max_retries: int = 3, timeout: int = 25) -> bool:
    """Download a single URL to a destination path with retries."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_path, "wb") as out:
                out.write(resp.read())
            os.replace(temp_path, dest_path)
            return True
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt == max_retries - 1:
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def download_splits(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """Download the official split CSV files."""
    local_splits = {}
    for split_key, relative_path in SPLIT_FILES.items():
        url = f"{GCS_BASE_URL}/{relative_path}"
        dest = os.path.join(data_dir, relative_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        success = download_url(url, dest)
        if success:
            local_splits[split_key] = dest
        else:
            print(f"[Warning] Could not download split CSV: {url}")
    return local_splits


def parse_split_csv(csv_path: str) -> list:
    """Read pairs of (s1_filename, label_filename) from a split CSV."""
    pairs = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            s1_file, lbl_file = row[0].strip(), row[1].strip()
            if s1_file.lower() in ("s1hand", "chip_id", "filename", "name"):
                continue
            pairs.append((s1_file, lbl_file))
    return pairs


def load_chip_ids_from_csv(csv_path: str) -> list:
    """Read list of chip IDs from a split CSV."""
    pairs = parse_split_csv(csv_path)
    return [s1.replace("_S1Hand.tif", "").replace(".tif", "") for s1, _ in pairs]



def download_sen1floods11(data_dir: str = DEFAULT_DATA_DIR,
                          include_perm: bool = True,
                          workers: int = 16) -> dict:
    """
    Download Sen1Floods11 hand-labeled S1 chips, LabelHand masks, and permanent water splits.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_splits = download_splits(data_dir)

    total_download_tasks = []

    # 1. Flood Hand-Labeled Splits
    s1_hand_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
    label_hand_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "LabelHand")
    os.makedirs(s1_hand_dir, exist_ok=True)
    os.makedirs(label_hand_dir, exist_ok=True)

    for split_key in ["train", "val", "test", "bolivia"]:
        if split_key not in local_splits:
            continue
        pairs = parse_split_csv(local_splits[split_key])
        for s1_f, lbl_f in pairs:
            s1_url = f"{GCS_BASE_URL}/data/flood_events/HandLabeled/S1Hand/{s1_f}"
            s1_dest = os.path.join(s1_hand_dir, s1_f)
            lbl_url = f"{GCS_BASE_URL}/data/flood_events/HandLabeled/LabelHand/{lbl_f}"
            lbl_dest = os.path.join(label_hand_dir, lbl_f)
            total_download_tasks.append((s1_url, s1_dest))
            total_download_tasks.append((lbl_url, lbl_dest))

    # 2. Permanent Water Splits (if requested)
    if include_perm:
        s1_perm_dir = os.path.join(data_dir, "v1.1", "data", "perm_water", "S1Perm")
        jrc_perm_dir = os.path.join(data_dir, "v1.1", "data", "perm_water", "JRCPerm")
        os.makedirs(s1_perm_dir, exist_ok=True)
        os.makedirs(jrc_perm_dir, exist_ok=True)

        for split_key in ["perm_train", "perm_val", "perm_test"]:
            if split_key not in local_splits:
                continue
            pairs = parse_split_csv(local_splits[split_key])
            for s1_f, lbl_f in pairs:
                s1_url = f"{GCS_BASE_URL}/data/perm_water/S1Perm/{s1_f}"
                s1_dest = os.path.join(s1_perm_dir, s1_f)
                lbl_url = f"{GCS_BASE_URL}/data/perm_water/JRCPerm/{lbl_f}"
                lbl_dest = os.path.join(jrc_perm_dir, lbl_f)
                total_download_tasks.append((s1_url, s1_dest))
                total_download_tasks.append((lbl_url, lbl_dest))

    print(f"\nTotal files to verify/download: {len(total_download_tasks)}")
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_url, url, dest): (url, dest) for url, dest in total_download_tasks}
        for fut in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == len(total_download_tasks):
                print(f"  Progress: {completed}/{len(total_download_tasks)} files ({(completed/len(total_download_tasks))*100:.1f}%)", end="\r")

    print(f"\nDownload completed in {time.time() - t0:.1f}s.")
    return local_splits


# ---------------------------------------------------------------------------
# Normalization & Preprocessing
# ---------------------------------------------------------------------------
def normalize_s1_band(band: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-1 SAR array to [0, 1].
    Handles both linear intensity (DN > 0) and backscatter in dB (< 0).
    """
    band = np.nan_to_num(band, nan=DB_MIN)
    if np.any(band > 1.0):
        # Linear power -> convert to dB
        band_db = 10.0 * np.log10(np.maximum(band, 1e-6))
    else:
        band_db = band

    norm = np.clip((band_db - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 Hand-Labeled and Permanent Water Sentinel-1 chips.

    Outputs:
        x : Tensor of shape (C, H, W), float32 normalized in [0, 1]
            - If in_channels == 2: [VV_norm, VH_norm]
            - If in_channels == 3: [VV_norm, VH_norm, (VH - VV + 1)/2 difference ratio]
        y : Tensor of shape (1, H, W), float32 (1.0 = water, 0.0 = dry, -1.0 = ignore)
        mask : Tensor of shape (1, H, W), float32 (1.0 = valid pixel, 0.0 = ignore pixel)
    """

    def __init__(self,
                 data_dir: str = DEFAULT_DATA_DIR,
                 split: str = "train",
                 in_channels: int = 2,
                 include_perm: bool = False,
                 augment: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.in_channels = in_channels
        self.augment = augment

        self.samples = []  # List of (s1_full_path, label_full_path)

        # 1. Load Hand-Labeled Flood split
        s1_hand_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
        label_hand_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "LabelHand")
        split_csv = os.path.join(data_dir, SPLIT_FILES.get(split, SPLIT_FILES["train"]))

        if os.path.exists(split_csv):
            pairs = parse_split_csv(split_csv)
            for s1_f, lbl_f in pairs:
                s1_p = os.path.join(s1_hand_dir, s1_f)
                lbl_p = os.path.join(label_hand_dir, lbl_f)
                if os.path.exists(s1_p) and os.path.exists(lbl_p):
                    self.samples.append((s1_p, lbl_p))

        # Also include Bolivia event in train/test if requested
        if split == "train":
            bolivia_csv = os.path.join(data_dir, SPLIT_FILES["bolivia"])
            if os.path.exists(bolivia_csv):
                for s1_f, lbl_f in parse_split_csv(bolivia_csv):
                    s1_p = os.path.join(s1_hand_dir, s1_f)
                    lbl_p = os.path.join(label_hand_dir, lbl_f)
                    if os.path.exists(s1_p) and os.path.exists(lbl_p):
                        self.samples.append((s1_p, lbl_p))

        # 2. Load Permanent Water split (if requested)
        if include_perm:
            perm_key = f"perm_{split}" if f"perm_{split}" in SPLIT_FILES else "perm_train"
            perm_csv = os.path.join(data_dir, SPLIT_FILES[perm_key])
            s1_perm_dir = os.path.join(data_dir, "v1.1", "data", "perm_water", "S1Perm")
            jrc_perm_dir = os.path.join(data_dir, "v1.1", "data", "perm_water", "JRCPerm")

            if os.path.exists(perm_csv):
                for s1_f, lbl_f in parse_split_csv(perm_csv):
                    s1_p = os.path.join(s1_perm_dir, s1_f)
                    lbl_p = os.path.join(jrc_perm_dir, lbl_f)
                    if os.path.exists(s1_p) and os.path.exists(lbl_p):
                        self.samples.append((s1_p, lbl_p))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s1_path, label_path = self.samples[idx]

        # Read S1 GeoTIFF (Band 1: VV, Band 2: VH)
        with rasterio.open(s1_path) as ds:
            vv = ds.read(1).astype(np.float32)
            vh = ds.read(2).astype(np.float32)

        # Read Label GeoTIFF (1 = water, 0 = dry, -1 = ignore)
        with rasterio.open(label_path) as ds:
            lbl = ds.read(1).astype(np.float32)

        # Align shapes if s1 and label differ slightly
        if vv.shape != lbl.shape:
            min_h = min(vv.shape[0], lbl.shape[0])
            min_w = min(vv.shape[1], lbl.shape[1])
            vv = vv[:min_h, :min_w]
            vh = vh[:min_h, :min_w]
            lbl = lbl[:min_h, :min_w]

        # Normalize SAR bands
        vv_norm = normalize_s1_band(vv)
        vh_norm = normalize_s1_band(vh)

        if self.in_channels == 3:
            # Add cross-polarization difference ratio
            diff_norm = np.clip((vh_norm - vv_norm + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
            x = np.stack([vv_norm, vh_norm, diff_norm], axis=0)
        else:
            x = np.stack([vv_norm, vh_norm], axis=0)

        # Binary label and valid mask (2D arrays)
        valid_mask = (lbl >= 0).astype(np.float32)
        y = np.where(lbl == 1, 1.0, 0.0).astype(np.float32)

        # Ensure exact 512x512 spatial dimensions for PyTorch batching
        target_H, target_W = 512, 512
        C, H, W = x.shape
        if H > target_H or W > target_W:
            sy = max((H - target_H) // 2, 0)
            sx = max((W - target_W) // 2, 0)
            x = x[:, sy:sy+target_H, sx:sx+target_W]
            y = y[sy:sy+target_H, sx:sx+target_W]
            valid_mask = valid_mask[sy:sy+target_H, sx:sx+target_W]

        C, H, W = x.shape
        if H < target_H or W < target_W:
            py = max(target_H - H, 0)
            px = max(target_W - W, 0)
            x = np.pad(x, ((0, 0), (0, py), (0, px)), mode="reflect")
            y = np.pad(y, ((0, py), (0, px)), mode="constant", constant_values=0)
            valid_mask = np.pad(valid_mask, ((0, py), (0, px)), mode="constant", constant_values=0)

        # Data augmentation for training
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, ::-1].copy()
                valid_mask = valid_mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[::-1, :].copy()
                valid_mask = valid_mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()
                valid_mask = np.rot90(valid_mask, k, axes=(0, 1)).copy()

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).unsqueeze(0).float(),
            torch.from_numpy(valid_mask).unsqueeze(0).float(),
        )


# ---------------------------------------------------------------------------
# CLI / Standalone Downloader & Tester
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download & Test Sen1Floods11 dataset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to store dataset")
    parser.add_argument("--download", action="store_true", help="Download Sen1Floods11 dataset")
    parser.add_argument("--include-perm", action="store_true", help="Include permanent water dataset")
    parser.add_argument("--workers", type=int, default=16, help="Number of download threads")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading Sen1Floods11 (include_perm={args.include_perm}) to {args.data_dir}...")
        download_sen1floods11(args.data_dir, include_perm=args.include_perm, workers=args.workers)

    # Test loading
    print("\nTesting Sen1Floods11 dataset loaders:")
    for split in ["train", "val", "test"]:
        try:
            ds = Sen1Floods11Dataset(data_dir=args.data_dir, split=split, in_channels=3, include_perm=args.include_perm)
            print(f"  {split.capitalize()} split: {len(ds)} chips available")
            if len(ds) > 0:
                x, y, mask = ds[0]
                print(f"    Sample 0: x shape={x.shape}, y shape={y.shape}, mask shape={mask.shape}")
                print(f"    Water pixels: {int(y.sum())}, Valid pixels: {int(mask.sum())}/{mask.numel()}")
        except Exception as e:
            print(f"  {split.capitalize()} split error: {e}")


if __name__ == "__main__":
    main()
