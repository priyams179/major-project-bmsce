"""
Dataset Loader for FloodNet UAV Imagery with Indian Flood Domain Adaptation.
Handles high-resolution aerial imagery and semantic segmentation masks.
Supports both multi-class damage mapping (10 classes) and binary flood extent mapping.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

try:
    from models.domain_adaptation import get_indian_flood_augmentation_pipeline
except ImportError:
    from flood_management_uav.models.domain_adaptation import get_indian_flood_augmentation_pipeline


class FloodNetSegmentationDataset(Dataset):
    """
    PyTorch Dataset for FloodNet UAV aerial segmentation.

    Classes:
      0: Background
      1: Building Flooded
      2: Building Non-Flooded
      3: Road Flooded
      4: Road Non-Flooded
      5: Water
      6: Tree
      7: Vehicle
      8: Pool
      9: Grass
    """

    CLASS_NAMES = [
        "Background",
        "Building Flooded",
        "Building Non-Flooded",
        "Road Flooded",
        "Road Non-Flooded",
        "Water",
        "Tree",
        "Vehicle",
        "Pool",
        "Grass",
    ]

    # Binary mapping: 1 (Flooded) = Water, Road Flooded, Building Flooded
    FLOOD_CLASSES = [1, 3, 5]

    def __init__(
        self,
        image_dir,
        mask_dir=None,
        transform=None,
        is_binary=False,
        apply_indian_adaptation=True,
        img_size=(512, 512),
        is_train=True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_binary = is_binary
        self.is_train = is_train
        self.img_size = img_size

        # Find all images
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        self.image_paths = sorted(list(set(self.image_paths)))

        if len(self.image_paths) == 0:
            # Look in subdirectories if present (e.g. train/images)
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
            self.image_paths = sorted(list(set(self.image_paths)))

        if transform is not None:
            self.transform = transform
        elif apply_indian_adaptation:
            self.transform = get_indian_flood_augmentation_pipeline(img_size=img_size, is_train=is_train)
        else:
            h, w = img_size
            if is_train:
                self.transform = A.Compose([
                    A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=h, width=w),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, img_path):
        """Finds matching mask file for a given image file."""
        if self.mask_dir is None:
            return None

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Check standard conventions: base_name.png, base_name_lab.png, base_name_mask.png
        candidates = [
            os.path.join(self.mask_dir, f"{base_name}.png"),
            os.path.join(self.mask_dir, f"{base_name}_lab.png"),
            os.path.join(self.mask_dir, f"{base_name}_mask.png"),
            os.path.join(self.mask_dir, f"{base_name}.jpg"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self._find_mask_path(img_path)
        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                # Align mask shape if downsampled or slightly mismatched
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            if self.is_binary:
                # Convert multi-class mask to binary flood mask:
                # 1 if pixel is in FLOOD_CLASSES (water, flooded road, flooded building) else 0
                binary_mask = np.isin(mask, self.FLOOD_CLASSES).astype(np.uint8)
                mask = binary_mask

            # Apply joint transforms
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Convert to PyTorch tensors
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask_t = torch.from_numpy(mask).long()

            return {
                "image": image_t,
                "mask": mask_t,
                "image_path": img_path,
                "mask_path": mask_path,
            }
        else:
            # Inference-only sample without mask
            augmented = self.transform(image=image)
            image = augmented["image"]
            image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()

            return {
                "image": image_t,
                "image_path": img_path,
            }
