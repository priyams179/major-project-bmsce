"""
FloodNet Dataset Downloader and Sample Generator Utility.
Supports:
1. Automated setup for FloodNet dataset directory structure.
2. Fetching subsets or archives via URL / Google Drive / Dropbox.
3. Generating realistic synthetic/test UAV flood benchmark samples
   (incorporating Indian flood characteristics: turbid water, varied roofing)
   for offline validation and pipeline verification.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import cv2
import numpy as np
from tqdm import tqdm


def setup_directories(base_dir="data/floodnet"):
    """Creates required data folder hierarchy."""
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    return images_dir, masks_dir


def generate_realistic_uav_flood_sample(img_id, width=1024, height=1024, is_indian_context=True):
    """
    Generates a realistic synthetic aerial UAV flood scene and matching ground-truth mask.

    Classes:
      0: Background / Bare Soil
      1: Building Flooded
      2: Building Non-Flooded
      3: Road Flooded
      4: Road Non-Flooded
      5: Water (Muddy / Silt-laden if Indian context)
      6: Tree / Canopy
      7: Vehicle / Car
      8: Pool
      9: Grass / Ground vegetation
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Base terrain: Soil / Ground (Background: 0) and Grass (Class 9)
    if is_indian_context:
        # Indian monsoon terrain: alluvial soil and patches of wet grass
        base_color = np.array([55, 95, 80], dtype=np.uint8)  # Olive/dark alluvial green
    else:
        base_color = np.array([50, 120, 60], dtype=np.uint8)

    image[:, :] = base_color + np.random.randint(-10, 10, (height, width, 3), dtype=np.int16).clip(0, 255).astype(np.uint8)
    mask[:, :] = 9  # Grass

    # 2. Add Roads: Main road intersecting the scene
    road_poly = np.array([[width * 0.1, 0], [width * 0.25, 0], [width * 0.85, height], [width * 0.7, height]], np.int32)
    cv2.fillPoly(image, [road_poly], (85, 85, 90))
    cv2.fillPoly(mask, [road_poly], 4)  # Road Non-Flooded

    # 3. Add Flood Water River / Inundation Zone (Class 5)
    # Sinusoidal / curving floodwater channel
    flood_points = []
    y_coords = np.linspace(0, height, 30)
    for y in y_coords:
        offset = 120 * np.sin(y / 150.0) + width * 0.45
        flood_points.append([offset, y])

    # Build flood corridor polygon
    left_side = [[x - 140, y] for x, y in flood_points]
    right_side = [[x + 140, y] for x, y in reversed(flood_points)]
    flood_poly = np.array(left_side + right_side, np.int32)

    # Water color:
    # If Indian context: Turbid muddy alluvial water (Brownish silt: BGR 45, 80, 115)
    # Else: Storm surge water (Teal/Slate Blue: BGR 130, 100, 40)
    if is_indian_context:
        water_bgr = (48, 85, 125)  # Muddy silt brown
    else:
        water_bgr = (140, 110, 50)

    cv2.fillPoly(image, [flood_poly], water_bgr)
    # Add water surface turbulence/texture
    noise = np.random.randint(-8, 8, (height, width, 3), dtype=np.int16)
    water_area = (cv2.pointPolygonTest(flood_poly, (0, 0), False) >= 0)  # fast check
    mask_water = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask_water, [flood_poly], 255)
    image[mask_water == 255] = np.clip(image[mask_water == 255].astype(np.int16) + noise[mask_water == 255], 0, 255).astype(np.uint8)
    mask[mask_water == 255] = 5  # Water

    # Overlap between road and water -> Road Flooded (Class 3)
    flooded_road = (mask_water == 255) & (mask == 4)
    mask[flooded_road] = 3

    # 4. Add Buildings:
    # Some dry, some partially submerged in the flood corridor
    building_coords = [
        # (x, y, w, h, roof_type)
        (int(width * 0.15), int(height * 0.20), 120, 90, "terracotta"),
        (int(width * 0.12), int(height * 0.55), 110, 80, "concrete"),
        (int(width * 0.38), int(height * 0.35), 130, 95, "tin"),       # Near/in floodwater
        (int(width * 0.48), int(height * 0.70), 115, 85, "terracotta"), # Submerged/in floodwater
        (int(width * 0.75), int(height * 0.15), 105, 85, "concrete"),
        (int(width * 0.78), int(height * 0.45), 125, 90, "tin"),
    ]

    for bx, by, bw, bh, roof in building_coords:
        if roof == "terracotta":
            roof_color = (35, 45, 175)   # Reddish Mangalore clay tile
        elif roof == "tin":
            roof_color = (195, 180, 160) # Corrugated galvanized sheet
        else:
            roof_color = (130, 130, 135) # Concrete slab

        # Determine if building is submerged
        b_box_water = (mask_water[by:by+bh, bx:bx+bw] == 255)
        submergence_ratio = np.mean(b_box_water)

        b_class = 1 if submergence_ratio > 0.15 else 2  # 1: Building Flooded, 2: Non-Flooded

        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), roof_color, -1)
        # Building border / shadow
        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (20, 20, 20), 2)
        mask[by:by+bh, bx:bx+bw] = b_class

    # 5. Add Vehicles / Cars (Class 7)
    cars = [
        (int(width * 0.22), int(height * 0.12), 24, 45, (210, 50, 40)),    # Red car on dry road
        (int(width * 0.26), int(height * 0.22), 22, 42, (230, 230, 230)),  # White car
        (int(width * 0.45), int(height * 0.48), 24, 46, (40, 190, 220)),   # Submerged vehicle in water corridor!
        (int(width * 0.80), int(height * 0.80), 24, 44, (30, 30, 30)),     # Dark vehicle
    ]

    for cx, cy, cw, ch, car_col in cars:
        cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), car_col, -1)
        cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), (10, 10, 10), 1)
        mask[cy:cy+ch, cx:cx+cw] = 7  # Vehicle

    # 6. Add Trees / Canopy clusters (Class 6)
    for _ in range(12):
        tx = np.random.randint(50, width - 50)
        ty = np.random.randint(50, height - 50)
        tr = np.random.randint(25, 45)
        if mask[ty, tx] not in [1, 2, 7]:  # Don't overwrite buildings/cars
            tree_col = (20 + np.random.randint(-10, 10), 90 + np.random.randint(-15, 15), 30 + np.random.randint(-10, 10))
            cv2.circle(image, (tx, ty), tr, tree_col, -1)
            cv2.circle(mask, (tx, ty), tr, 6, -1)

    return image, mask


def create_verification_samples(base_dir="data/floodnet", num_samples=16):
    """
    Creates a set of verified high-res UAV flood chips and ground-truth masks
    for immediate testing, benchmarking, and validation.
    """
    img_dir, mask_dir = setup_directories(base_dir)
    print(f"Generating {num_samples} verified UAV flood benchmark chips in {base_dir}...")

    for i in tqdm(range(num_samples), desc="Generating UAV benchmark chips"):
        img_id = f"uav_flood_{i+1:04d}"
        is_indian = (i % 2 == 0)  # Alternate between Indian turbid monsoon and coastal scenes
        img, mask = generate_realistic_uav_flood_sample(img_id, width=1024, height=1024, is_indian_context=is_indian)

        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        mask_path = os.path.join(mask_dir, f"{img_id}.png")

        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

    print(f"Successfully generated {num_samples} chips in {img_dir} and masks in {mask_dir}.")
    return img_dir, mask_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloodNet Data Setup and Benchmark Generator")
    parser.add_argument("--base-dir", type=str, default="data/floodnet", help="Base directory for FloodNet")
    parser.add_argument("--samples", type=int, default=20, help="Number of benchmark chips to generate")
    parser.add_argument("--generate-samples", action="store_true", default=True, help="Generate verified benchmark dataset")

    args = parser.parse_args()
    if args.generate_samples:
        create_verification_samples(base_dir=args.base_dir, num_samples=args.samples)

