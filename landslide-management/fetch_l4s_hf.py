#!/usr/bin/env python3
"""
Direct Multi-Threaded Landslide4Sense Downloader from Hugging Face

Fetches benchmark patches from `ibm-nasa-geospatial/Landslide4sense`,
extracts the 14-band images and binary landslide masks, and saves them
directly into standard `data/processed/l4s/{train,val,test}/` format
ready for training with `train_segmentation.py`.
"""

import argparse
import concurrent.futures
import io
import json
import os
import sys
import urllib.request
import h5py
import numpy as np
from tqdm import tqdm

BASE_HF_URL = "https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense/resolve/main"

def download_patch(split: str, idx: int, out_dir: str) -> bool:
    """Download single image and mask, convert and save as .npy."""
    img_name = f"image_{idx}.h5"
    mask_name = f"mask_{idx}.h5"

    img_url = f"{BASE_HF_URL}/images/{split}/{img_name}"
    mask_url = f"{BASE_HF_URL}/annotations/{split}/{mask_name}"

    try:
        # Download image
        req = urllib.request.Request(img_url, headers={"User-Agent": "LandslideAI/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            img_buf = io.BytesIO(resp.read())

        with h5py.File(img_buf, "r") as f:
            # (128, 128, 14) -> transpose to (14, 128, 128)
            img_arr = f["img"][:].astype(np.float32)
            if img_arr.ndim == 3 and img_arr.shape[-1] == 14:
                img_arr = np.transpose(img_arr, (2, 0, 1))

        # Download mask
        req_m = urllib.request.Request(mask_url, headers={"User-Agent": "LandslideAI/1.0"})
        with urllib.request.urlopen(req_m, timeout=20) as resp_m:
            mask_buf = io.BytesIO(resp_m.read())

        with h5py.File(mask_buf, "r") as f_m:
            mask_arr = f_m["mask"][:].astype(np.float32)

        # Valid mask: 1.0 everywhere except where mask == -1 or NaN
        valid_mask = np.ones_like(mask_arr, dtype=np.float32)
        valid_mask[mask_arr < 0] = 0.0
        mask_arr[mask_arr < 0] = 0.0

        # Save as npy
        out_img = os.path.join(out_dir, f"image_{idx:05d}.npy")
        out_mask = os.path.join(out_dir, f"mask_{idx:05d}.npy")
        out_valid = os.path.join(out_dir, f"valid_{idx:05d}.npy")

        np.save(out_img, img_arr)
        np.save(out_mask, mask_arr)
        np.save(out_valid, valid_mask)
        return True
    except Exception as e:
        return False


def fetch_split(split: str, count: int, root_dir: str, max_workers: int = 12) -> int:
    """Download and process split."""
    split_dir = os.path.join(root_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    print(f"\n--- Downloading {count} patches for split '{split}' into {split_dir} ---")

    # Probe available files
    api_url = f"https://huggingface.co/api/datasets/ibm-nasa-geospatial/Landslide4sense/tree/main/images/{split}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        files = json.loads(r.read())

    # Extract valid indices
    indices = []
    for f in files:
        fname = os.path.basename(f.get("path", ""))
        if fname.startswith("image_") and fname.endswith(".h5"):
            try:
                num = int(fname.replace("image_", "").replace(".h5", ""))
                indices.append(num)
            except ValueError:
                pass

    indices = sorted(indices)[:count]
    print(f"Found {len(indices)} patches to download.")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_patch, split, idx, split_dir): idx for idx in indices}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{split}"):
            if fut.result():
                success_count += 1

    print(f"Successfully processed {success_count}/{len(indices)} patches for '{split}'.")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Landslide4Sense from Hugging Face")
    parser.add_argument("--train-count", type=int, default=300, help="Number of training patches (default: 300)")
    parser.add_argument("--val-count", type=int, default=50, help="Number of validation patches (default: 50)")
    parser.add_argument("--test-count", type=int, default=50, help="Number of test patches (default: 50)")
    parser.add_argument("--output-dir", default="data/processed/l4s", help="Output directory")
    parser.add_argument("--workers", type=int, default=12, help="Download threads")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_cnt = fetch_split("train", args.train_count, args.output_dir, args.workers)
    v_cnt = fetch_split("validation", args.val_count, args.output_dir, args.workers)
    te_cnt = fetch_split("test", args.test_count, args.output_dir, args.workers)

    # In L4S on HF, 'validation' is used, rename or link to 'val' for consistency
    val_dir = os.path.join(args.output_dir, "val")
    validation_dir = os.path.join(args.output_dir, "validation")
    if os.path.exists(validation_dir) and not os.path.exists(val_dir):
        os.rename(validation_dir, val_dir)

    print("\n" + "=" * 70)
    print("Landslide4Sense Benchmark Dataset Preparation Complete!")
    print(f"  Train samples: {t_cnt}")
    print(f"  Val samples:   {v_cnt}")
    print(f"  Test samples:  {te_cnt}")
    print(f"Location: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

