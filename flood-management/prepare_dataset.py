#!/usr/bin/env python3
"""
Prepare dataset for CNN-based flood detection.

Loads the speckle-filtered before/after dB images and the flood map label,
normalizes them, and saves as .npy files.  Generates patch coordinate lists
for train / val / test splits using a geographic split (columns).

Usage:
    .venv/bin/python prepare_dataset.py
"""

import json
import os
import time

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = "output"
OUTPUT_DIR = "dataset"
PATCH_SIZE = 256
STRIDE = 256       # no overlap for training patches
DB_MIN = -30.0     # clip dB values
DB_MAX = 0.0
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def normalize_db(img):
    """Clip dB to [DB_MIN, DB_MAX] and scale to [0, 1]."""
    return np.clip((img - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0).astype(np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # --- Load images ---
    print("Loading images...")
    with rasterio.open(os.path.join(INPUT_DIR, "before_filtered_dB.tif")) as ds:
        before = ds.read(1).astype(np.float32)
    with rasterio.open(os.path.join(INPUT_DIR, "after_filtered_dB.tif")) as ds:
        after = ds.read(1).astype(np.float32)
    with rasterio.open(os.path.join(INPUT_DIR, "flood_map.tif")) as ds:
        flood = ds.read(1).astype(np.float32)

    H, W = before.shape
    print(f"  Image size: {W} x {H}")

    # --- Normalize ---
    print("Normalizing...")
    before_n = normalize_db(before)
    after_n = normalize_db(after)
    del before, after

    # Stack: (2, H, W)
    X = np.stack([before_n, after_n], axis=0)
    del before_n, after_n

    # Binary label: (H, W)
    y = (flood > 0.5).astype(np.float32)
    del flood

    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  Flood pixel ratio: {y.mean():.4f}")

    # --- Save full images as memmap ---
    print("Saving normalized images (memmap)...")
    X_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    np.save(X_path, X)
    np.save(y_path, y)
    del X, y

    # --- Generate patch coordinates ---
    print("Generating patch coordinates...")
    # X is axis 1 (width), split by columns
    split_boundaries = {}
    cum_ratio = 0.0
    for name, ratio in SPLIT_RATIOS.items():
        start = int(W * cum_ratio)
        end = int(W * (cum_ratio + ratio))
        split_boundaries[name] = (start, end)
        cum_ratio += ratio
    # Make sure last split reaches W
    last_name = list(SPLIT_RATIOS.keys())[-1]
    split_boundaries[last_name] = (split_boundaries[last_name][0], W)

    patches = {"train": [], "val": [], "test": []}
    for py in range(0, H - PATCH_SIZE + 1, STRIDE):
        for px in range(0, W - PATCH_SIZE + 1, STRIDE):
            center_x = px + PATCH_SIZE // 2
            for name, (start, end) in split_boundaries.items():
                if start <= center_x < end:
                    patches[name].append((py, px))
                    break

    for name in patches:
        print(f"  {name}: {len(patches[name])} patches")

    # --- Save patch coordinates ---
    with open(os.path.join(OUTPUT_DIR, "splits.json"), "w") as f:
        json.dump(patches, f)

    # --- Stats ---
    total = sum(len(p) for p in patches.values())
    flood_ratio = float(np.load(y_path, mmap_mode="r").mean())
    print(f"\nDataset ready:")
    print(f"  Total patches: {total}")
    print(f"  Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"  Flood pixel ratio: {flood_ratio:.4f}")
    print(f"  Saved to: {OUTPUT_DIR}/")
    print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
