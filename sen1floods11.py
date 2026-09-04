#!/usr/bin/env python3
"""
Sen1Floods11 Dataset Integration

Provides automated downloading and PyTorch Dataset / DataLoader classes for
the Sen1Floods11 benchmark dataset (v1.1 hand-labeled Sentinel-1 SAR imagery).

Sen1Floods11 specifications:
  - 446 hand-labeled 512x512 chips across 11 global flood events
  - Sentinel-1 SAR: Band 1 = VV, Band 2 = VH
  - Ground truth labels: 0 = not flooded, 1 = flooded, -1 = ignore (cloud / nodata)
  - Official splits: 252 train, 89 validation, 90 test chips

Usage:
    uv run python sen1floods11.py --download --data-dir data/sen1floods11
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
}

# dB Normalization range for Sentinel-1
DB_MIN = -35.0
DB_MAX = 0.0


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------
def download_url(url: str, dest_path: str, max_retries: int = 3, timeout: int = 20) -> bool:
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
                print(f"[Error] Failed to download {url}: {e}", file=sys.stderr)
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def download_splits(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """Download the official split CSV files."""
    splits_dir = os.path.join(data_dir, "splits", "flood_handlabeled")
    os.makedirs(splits_dir, exist_ok=True)

    local_splits = {}
    for split_name, relative_path in SPLIT_FILES.items():
        url = f"{GCS_BASE_URL}/{relative_path}"
        dest = os.path.join(data_dir, relative_path)
        print(f"Downloading split file: {split_name}...")
        success = download_url(url, dest)
        if not success:
            raise RuntimeError(f"Could not download split CSV: {url}")
        local_splits[split_name] = dest

    return local_splits


def load_chip_ids_from_csv(csv_path: str) -> list:
    """Read chip IDs / filenames from a Sen1Floods11 split CSV."""
    chip_ids = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            item = row[0].strip()
            # If CSV has header or empty rows
            if not item or item.lower() in ("s1hand", "chip_id", "filename", "name"):
                continue
            # Strip suffixes if present
            chip_id = item.replace("_S1Hand.tif", "").replace("_LabelHand.tif", "").replace(".tif", "")
            chip_ids.append(chip_id)
    return chip_ids


def download_sen1floods11(data_dir: str = DEFAULT_DATA_DIR,
                          splits: list = ("train", "val", "test"),
                          max_chips: int = 0,
                          workers: int = 8) -> dict:
    """
    Download Sen1Floods11 hand-labeled S1 chips, LabelHand masks, and split CSVs.
    """
    os.makedirs(data_dir, exist_ok=True)
    local_splits = download_splits(data_dir)

    all_chips = {}
    total_download_tasks = []

    s1_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
    label_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "LabelHand")
    os.makedirs(s1_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for split_name in splits:
        csv_file = local_splits[split_name]
        chip_ids = load_chip_ids_from_csv(csv_file)
        if max_chips > 0:
            chip_ids = chip_ids[:max_chips]
        all_chips[split_name] = chip_ids

        for cid in chip_ids:
            s1_url = f"{GCS_BASE_URL}/data/flood_events/HandLabeled/S1Hand/{cid}_S1Hand.tif"
            s1_dest = os.path.join(s1_dir, f"{cid}_S1Hand.tif")
            label_url = f"{GCS_BASE_URL}/data/flood_events/HandLabeled/LabelHand/{cid}_LabelHand.tif"
            label_dest = os.path.join(label_dir, f"{cid}_LabelHand.tif")

            total_download_tasks.append((s1_url, s1_dest))
            total_download_tasks.append((label_url, label_dest))

    print(f"\nTotal files to verify/download: {len(total_download_tasks)}")
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_url, url, dest): (url, dest) for url, dest in total_download_tasks}
        for fut in as_completed(futures):
            url, dest = futures[fut]
            try:
                res = fut.result()
                if not res:
                    print(f"[Warning] Failed file: {url}")
            except Exception as exc:
                print(f"[Warning] Exception downloading {url}: {exc}")
            completed += 1
            if completed % 20 == 0 or completed == len(total_download_tasks):
                print(f"  Progress: {completed}/{len(total_download_tasks)} files ({(completed/len(total_download_tasks))*100:.1f}%)", end="\r")

    print(f"\nDownload completed in {time.time() - t0:.1f}s.")
    return all_chips


# ---------------------------------------------------------------------------
# Normalization & Preprocessing
# ---------------------------------------------------------------------------
def normalize_s1_band(band: np.ndarray) -> np.ndarray:
    """
    Normalize Sentinel-1 SAR array to [0, 1].
    Handles both linear intensity (DN > 0) and backscatter in dB (< 0).
    """
    band = np.nan_to_num(band, nan=DB_MIN)
    # Check if band is in linear power or already in dB
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
    PyTorch Dataset for Sen1Floods11 Hand-Labeled Sentinel-1 chips.

    Outputs:
        x : Tensor of shape (C, H, W), float32 normalized in [0, 1]
            - If in_channels == 2: [VV_norm, VH_norm]
            - If in_channels == 3: [VV_norm, VH_norm, VH_norm - VV_norm + 0.5]
        y : Tensor of shape (1, H, W), float32 (1.0 = flood, 0.0 = dry, -1.0 = ignore)
        mask : Tensor of shape (1, H, W), float32 (1.0 = valid pixel, 0.0 = ignore pixel)
    """

    def __init__(self,
                 data_dir: str = DEFAULT_DATA_DIR,
                 split: str = "train",
                 in_channels: int = 2,
                 augment: bool = False,
                 chip_ids: list = None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.in_channels = in_channels
        self.augment = augment

        self.s1_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "S1Hand")
        self.label_dir = os.path.join(data_dir, "v1.1", "data", "flood_events", "HandLabeled", "LabelHand")

        if chip_ids is not None:
            self.chip_ids = chip_ids
        else:
            split_csv = os.path.join(data_dir, SPLIT_FILES[split])
            if not os.path.exists(split_csv):
                raise FileNotFoundError(
                    f"Split CSV not found at {split_csv}. Please run with --download first."
                )
            self.chip_ids = load_chip_ids_from_csv(split_csv)

        # Filter out chips that might be missing locally
        valid_chips = []
        for cid in self.chip_ids:
            s1_p = os.path.join(self.s1_dir, f"{cid}_S1Hand.tif")
            lbl_p = os.path.join(self.label_dir, f"{cid}_LabelHand.tif")
            if os.path.exists(s1_p) and os.path.exists(lbl_p):
                valid_chips.append(cid)

        self.chip_ids = valid_chips

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, idx: int):
        cid = self.chip_ids[idx]
        s1_path = os.path.join(self.s1_dir, f"{cid}_S1Hand.tif")
        label_path = os.path.join(self.label_dir, f"{cid}_LabelHand.tif")

        # Read 2-band S1 GeoTIFF (Band 1: VV, Band 2: VH)
        with rasterio.open(s1_path) as ds:
            vv = ds.read(1).astype(np.float32)
            vh = ds.read(2).astype(np.float32)

        # Read 1-band Label GeoTIFF (0 = dry, 1 = flood, -1 = ignore)
        with rasterio.open(label_path) as ds:
            lbl = ds.read(1).astype(np.float32)

        # Normalize SAR
        vv_norm = normalize_s1_band(vv)
        vh_norm = normalize_s1_band(vh)

        if self.in_channels == 3:
            # Add cross-polarization difference channel
            diff_norm = np.clip((vh_norm - vv_norm + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
            x = np.stack([vv_norm, vh_norm, diff_norm], axis=0)
        else:
            x = np.stack([vv_norm, vh_norm], axis=0)

        # Binary label and valid mask
        # Label contains 1 (flood), 0 (dry), -1 (cloud/nodata)
        valid_mask = (lbl >= 0).astype(np.float32)
        y = np.where(lbl == 1, 1.0, 0.0).astype(np.float32)

        # Data augmentation
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
            torch.from_numpy(x),
            torch.from_numpy(y).unsqueeze(0),
            torch.from_numpy(valid_mask).unsqueeze(0),
        )


# ---------------------------------------------------------------------------
# CLI / Standalone Test
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download & Test Sen1Floods11 dataset")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory to store dataset")
    parser.add_argument("--download", action="store_true", help="Download Sen1Floods11 dataset")
    parser.add_argument("--chips", type=int, default=0, help="Max chips to download per split (0 for all)")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading Sen1Floods11 to {args.data_dir}...")
        download_sen1floods11(args.data_dir, max_chips=args.chips, workers=args.workers)

    # Test loading
    print("\nTesting Sen1Floods11 dataset loaders:")
    for split in ["train", "val", "test"]:
        try:
            ds = Sen1Floods11Dataset(data_dir=args.data_dir, split=split, in_channels=2, augment=(split == "train"))
            print(f"  {split.capitalize()} split: {len(ds)} chips available")
            if len(ds) > 0:
                x, y, mask = ds[0]
                print(f"    Sample 0: x shape={x.shape}, y shape={y.shape}, mask shape={mask.shape}")
                print(f"    Flood pixels: {int(y.sum())}, Valid pixels: {int(mask.sum())}/{mask.numel()}")
        except Exception as e:
            print(f"  {split.capitalize()} split error: {e}")


if __name__ == "__main__":
    main()

