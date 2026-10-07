"""
Extracts Bounding Box Annotations for Buildings and Vehicles from FloodNet Semantic Masks.
Converts mask contours to standardized YOLO format:
  class_id x_center y_center width height  (all normalized to [0, 1])

Classes:
  0: Building-Flooded
  1: Building-Non-Flooded
  2: Vehicle
"""

import os
import glob
import shutil
import cv2
import numpy as np
import yaml
from tqdm import tqdm


FLOODNET_TO_YOLO_UNIFIED = {
    1: 0,  # Building Flooded -> Building
    2: 0,  # Building Non-Flooded -> Building
    7: 1,  # Vehicle
}
YOLO_CLASS_NAMES_UNIFIED = ["Building", "Vehicle"]

FLOODNET_TO_YOLO_DETAILED = {
    1: 0,  # Building-Flooded
    2: 1,  # Building-Non-Flooded
    7: 2,  # Vehicle
}
YOLO_CLASS_NAMES_DETAILED = ["Building-Flooded", "Building-Non-Flooded", "Vehicle"]


def mask_to_yolo_boxes(mask, min_area=30, unified_buildings=True):
    """
    Extracts bounding boxes for target physical classes from a segmentation mask.

    Args:
        mask: 2D numpy array containing class IDs
        min_area: Minimum bounding box area in pixels to eliminate noise
        unified_buildings: If True, maps all buildings to class 0 ('Building') for pure physical asset detection.

    Returns:
        List of tuples: (yolo_class_id, x_center, y_center, width, height) normalized to [0, 1]
    """
    h, w = mask.shape
    boxes = []

    mapping = FLOODNET_TO_YOLO_UNIFIED if unified_buildings else FLOODNET_TO_YOLO_DETAILED

    # Process each target class group
    unique_target_yolo_ids = sorted(list(set(mapping.values())))
    for yolo_class_id in unique_target_yolo_ids:
        # Source mask classes that map to this YOLO class
        src_mask_classes = [src for src, tgt in mapping.items() if tgt == yolo_class_id]

        binary = np.zeros((h, w), dtype=np.uint8)
        for sc in src_mask_classes:
            binary[mask == sc] = 255

        if not np.any(binary):
            continue

        # Find contours using OpenCV
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Filter extremely thin edge artifacts
            if bw < 4 or bh < 4:
                continue

            # Compute normalized center and dimensions
            x_center = (bx + bw / 2.0) / w
            y_center = (by + bh / 2.0) / h
            norm_w = bw / float(w)
            norm_h = bh / float(h)

            # Clamp coordinates safely in [0.0, 1.0]
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            norm_w = min(max(norm_w, 0.0), 1.0)
            norm_h = min(max(norm_h, 0.0), 1.0)

            boxes.append((yolo_class_id, x_center, y_center, norm_w, norm_h))

    return boxes


def prepare_yolo_dataset(
    image_dir,
    mask_dir,
    output_yolo_dir="data/yolo_detection",
    val_split=0.2,
    min_area=30,
    copy_images=False,
    unified_buildings=True,
):
    """
    Creates standard YOLO dataset structure:
      output_yolo_dir/
        images/train, images/val
        labels/train, labels/val
        floodnet_detect.yaml
    """
    os.makedirs(os.path.join(output_yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "labels", "val"), exist_ok=True)

    # Find paired images and masks
    img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
    image_paths = sorted(list(set(image_paths)))

    print(f"Found {len(image_paths)} images to process for YOLO bounding box dataset.")

    valid_pairs = []
    for img_p in image_paths:
        base = os.path.splitext(os.path.basename(img_p))[0]
        # Candidate mask paths
        mask_candidates = [
            os.path.join(mask_dir, f"{base}.png"),
            os.path.join(mask_dir, f"{base}_lab.png"),
            os.path.join(mask_dir, f"{base}_mask.png"),
            os.path.join(mask_dir, f"{base}.jpg"),
        ]
        found_mask = None
        for cand in mask_candidates:
            if os.path.exists(cand):
                found_mask = cand
                break
        if found_mask is not None:
            valid_pairs.append((img_p, found_mask, base))

    print(f"Matched {len(valid_pairs)} valid image-mask pairs.")

    # Train / Val Split
    np.random.seed(42)
    indices = np.arange(len(valid_pairs))
    np.random.shuffle(indices)

    val_count = int(len(valid_pairs) * val_split)
    val_indices = set(indices[:val_count])

    class_names = YOLO_CLASS_NAMES_UNIFIED if unified_buildings else YOLO_CLASS_NAMES_DETAILED
    stats = {"train_samples": 0, "val_samples": 0, "total_boxes": 0, "class_counts": {i: 0 for i in range(len(class_names))}}

    for i, (img_p, mask_p, base) in enumerate(tqdm(valid_pairs, desc="Converting masks to YOLO bboxes")):
        split = "val" if i in val_indices else "train"
        if split == "train":
            stats["train_samples"] += 1
        else:
            stats["val_samples"] += 1

        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        boxes = mask_to_yolo_boxes(mask, min_area=min_area, unified_buildings=unified_buildings)
        stats["total_boxes"] += len(boxes)
        for b in boxes:
            stats["class_counts"][b[0]] = stats["class_counts"].get(b[0], 0) + 1

        # Write label text file
        label_dest = os.path.join(output_yolo_dir, "labels", split, f"{base}.txt")
        with open(label_dest, "w") as f:
            for b in boxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        # Either copy or symlink image
        ext = os.path.splitext(img_p)[1]
        img_dest = os.path.join(output_yolo_dir, "images", split, f"{base}{ext}")
        if copy_images:
            if not os.path.exists(img_dest):
                shutil.copy2(img_p, img_dest)
        else:
            # Create symlink if not already present
            if not os.path.exists(img_dest):
                os.symlink(os.path.abspath(img_p), img_dest)

    # Generate YOLO dataset YAML
    abs_yolo_dir = os.path.abspath(output_yolo_dir)
    yaml_names = {i: name for i, name in enumerate(class_names)}
    yaml_content = {
        "path": abs_yolo_dir,
        "train": "images/train",
        "val": "images/val",
        "names": yaml_names,
    }

    yaml_path = os.path.join(output_yolo_dir, "floodnet_detect.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("\n=== YOLO Dataset Generation Summary ===")
    print(f"YOLO Data Directory: {abs_yolo_dir}")
    print(f"Train Images: {stats['train_samples']}, Val Images: {stats['val_samples']}")
    print(f"Total Bounding Boxes: {stats['total_boxes']}")
    for cid, name in enumerate(class_names):
        print(f"  - {name}: {stats['class_counts'].get(cid, 0)} instances")
    print(f"Configuration YAML created at: {yaml_path}")

    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract YOLO bounding boxes from FloodNet masks")
    parser.add_argument("--image-dir", type=str, default="data/floodnet/images", help="Path to UAV images")
    parser.add_argument("--mask-dir", type=str, default="data/floodnet/masks", help="Path to semantic masks")
    parser.add_argument("--output-dir", type=str, default="data/yolo_detection", help="Target YOLO directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set fraction")
    parser.add_argument("--min-area", type=int, default=30, help="Minimum contour pixel area")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--detailed", action="store_true", help="Keep separate flooded/non-flooded building classes")

    args = parser.parse_args()
    if os.path.exists(args.image_dir) and os.path.exists(args.mask_dir):
        prepare_yolo_dataset(
            args.image_dir,
            args.mask_dir,
            output_yolo_dir=args.output_dir,
            val_split=args.val_split,
            min_area=args.min_area,
            copy_images=args.copy,
            unified_buildings=not args.detailed,
        )
    else:
        print(f"Image or mask directory does not exist yet ({args.image_dir}, {args.mask_dir}).")
        print("Run download_floodnet.py first or point to valid paths.")

