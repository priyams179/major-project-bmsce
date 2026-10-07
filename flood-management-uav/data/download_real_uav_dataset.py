"""
Fast multi-threaded downloader for real post-disaster UAV drone flood imagery and masks.
Sources real high-resolution aerial disaster flight data (1024x1024) from NOAA UAV missions.
Maps classes to standard flood disaster taxonomy:
  0: Background / Land
  1: Building Flooded / Damaged
  2: Building Non-Flooded / Intact
  3: Road Flooded
  4: Road Non-Flooded
  5: Water / Floodwater
  6: Tree / Canopy
  7: Vehicle
"""

import os
import urllib.request
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


FWISD_TO_STANDARD = {
    0: 0,
    1: 5,
    2: 6,
    3: 4,
    4: 3,
    5: 2,
    6: 1,
    7: 2,
    8: 1,
    9: 7,
    10: 7,
    11: 5,
}


def download_single_pair(sid, img_dir, mask_dir, img_url, ann_url):
    img_out = os.path.join(img_dir, sid)
    mask_out = os.path.join(mask_dir, sid)

    if os.path.exists(img_out) and os.path.exists(mask_out) and os.path.getsize(img_out) > 5000:
        return sid, True

    for attempt in range(3):
        try:
            req_img = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
            req_ann = urllib.request.Request(ann_url, headers={"User-Agent": "Mozilla/5.0"})

            with urllib.request.urlopen(req_img, timeout=20) as resp, open(img_out, "wb") as f:
                f.write(resp.read())

            with urllib.request.urlopen(req_ann, timeout=20) as resp, open(mask_out, "wb") as f:
                f.write(resp.read())

            # Remap mask to standard classes
            raw_mask = cv2.imread(mask_out, cv2.IMREAD_UNCHANGED)
            if raw_mask is not None:
                std_mask = np.zeros_like(raw_mask, dtype=np.uint8)
                for src_c, tgt_c in FWISD_TO_STANDARD.items():
                    std_mask[raw_mask == src_c] = tgt_c
                cv2.imwrite(mask_out, std_mask)
                return sid, True
        except Exception:
            if attempt == 2:
                if os.path.exists(img_out):
                    try: os.remove(img_out)
                    except OSError: pass
                if os.path.exists(mask_out):
                    try: os.remove(mask_out)
                    except OSError: pass
                return sid, False
    return sid, False


def download_real_uav_flood_dataset(dest_dir="data/real_flood_uav_large", num_samples=1500, max_workers=16):
    """
    Downloads paired real UAV disaster images and annotations from FWISD train & val splits.
    Supports scaling up to 1,980 authentic disaster drone flights.
    """
    img_dir = os.path.join(dest_dir, "images")
    mask_dir = os.path.join(dest_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    base_hf = "https://huggingface.co/datasets/kevinxue112/FWISD/resolve/main"

    tasks = []
    # Train split has 1000 samples
    train_count = min(num_samples, 1000)
    for i in range(1, train_count + 1):
        sid = f"train_{i:05d}.png"
        raw_sid = f"{i:05d}.png"
        img_url = f"{base_hf}/img_dir/train/{raw_sid}"
        ann_url = f"{base_hf}/ann_dir/train/{raw_sid}"
        tasks.append((sid, img_url, ann_url))

    # Val split has 980 samples
    if num_samples > 1000:
        val_count = min(num_samples - 1000, 980)
        for i in range(1, val_count + 1):
            sid = f"val_{i:05d}.png"
            raw_sid = f"{i:05d}.png"
            img_url = f"{base_hf}/img_dir/val/{raw_sid}"
            ann_url = f"{base_hf}/ann_dir/val/{raw_sid}"
            tasks.append((sid, img_url, ann_url))

    print(f"Parallel downloading {len(tasks)} real disaster UAV pairs ({max_workers} threads) into {dest_dir}...")
    downloaded = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_single_pair, sid, img_dir, mask_dir, img_url, ann_url
            ): sid
            for (sid, img_url, ann_url) in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching scaled disaster pairs"):
            sid, success = future.result()
            if success:
                downloaded += 1

    print(f"\nSuccessfully prepared {downloaded} real UAV disaster flight pairs in {dest_dir}!")
    return img_dir, mask_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--dest-dir", type=str, default="data/real_flood_uav_large")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    download_real_uav_flood_dataset(dest_dir=args.dest_dir, num_samples=args.num_samples, max_workers=args.workers)
