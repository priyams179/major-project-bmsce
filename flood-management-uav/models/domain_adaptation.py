"""
Domain Adaptation Module for Indian Flood-Prone Regions.

Indian flood scenarios (e.g., Brahmaputra in Assam, Kosi/Ganga in Bihar,
Western Ghats in Kerala, and urban monsoon inundations in Bengaluru/Chennai/Mumbai)
differ significantly from Western datasets like FloodNet (Hurricane Harvey):
1. Turbid / Muddy Water: Indian floodwater carries heavy alluvial silt, clay,
   and sediment, appearing opaque brown/yellowish rather than clear or blue-green.
2. Heterogeneous Roofing: Rural and peri-urban Indian settlements feature
   terracotta tiles (Mangalore tiles), corrugated galvanized iron/tin roofs,
   and unpainted flat RCC concrete terraces with water tanks.
3. Monsoon Precipitation & Haze: Reduced atmospheric visibility, intense rain
   streaks, and heavy overcast lighting.

This module provides differentiable and PIL/OpenCV-based augmentation transforms
specifically designed to bridge this domain gap using Albumentations.
"""

import random
import cv2
import numpy as np
import albumentations as A


def apply_turbid_muddy_water(image, **kwargs):
    """
    Simulates turbid, silt-heavy floodwater characteristic of Indian monsoon floods.
    Shifts lower-luminance / water-like regions toward silt brown/yellow tones.
    """
    if random.random() > 0.6:
        return image

    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

    alpha = random.uniform(0.4, 0.75)
    target_hue = random.uniform(15.0, 30.0)  # Silt brown / mud in OpenCV HSV (0-180 scale)

    # Detect water-like hues (blue/cyan/dark greens) or apply runoff tinting
    water_like = (h >= 35) & (h <= 135)
    if np.any(water_like):
        h[water_like] = (1 - alpha) * h[water_like] + alpha * target_hue
        s[water_like] = np.clip(s[water_like] * 1.15, 0, 255)
        v[water_like] = np.clip(v[water_like] * 0.90, 0, 255)

    img_hsv = np.stack([h, s, v], axis=-1).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def apply_indian_roof_augment(image, **kwargs):
    """
    Augments roof signatures to represent Indian housing diversity:
    - Terracotta Mangalore clay tiles (Red-orange boost)
    - Corrugated metal/tin sheets (Specular reflectance & silver-blue sheen)
    - Weathered concrete RCC flat roofs (Desaturated matte grey)
    """
    if random.random() > 0.5:
        return image

    img_aug = image.copy().astype(np.float32)
    style = random.choice(["terracotta", "corrugated_tin", "weathered_concrete"])

    if style == "terracotta":
        img_aug[:, :, 0] = np.clip(img_aug[:, :, 0] * random.uniform(1.08, 1.25), 0, 255)
        img_aug[:, :, 1] = np.clip(img_aug[:, :, 1] * random.uniform(0.90, 0.98), 0, 255)
    elif style == "corrugated_tin":
        img_aug[:, :, 2] = np.clip(img_aug[:, :, 2] * random.uniform(1.05, 1.20), 0, 255)
        img_aug = np.clip(img_aug * random.uniform(1.02, 1.12), 0, 255)
    elif style == "weathered_concrete":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        factor = random.uniform(0.15, 0.30)
        for c in range(3):
            img_aug[:, :, c] = (1 - factor) * img_aug[:, :, c] + factor * gray

    return img_aug.astype(np.uint8)


def apply_monsoon_atmosphere(image, **kwargs):
    """
    Simulates overcast monsoon lighting, high humidity haze, and diffuse shadows.
    """
    if random.random() > 0.4:
        return image

    h, w, c = image.shape
    img_aug = image.astype(np.float32)

    # Contrast dampening from overcast diffuse cloud cover
    mean_val = np.mean(img_aug, axis=(0, 1), keepdims=True)
    contrast_factor = random.uniform(0.80, 0.95)
    img_aug = mean_val + contrast_factor * (img_aug - mean_val)

    # Atmospheric haze
    haze = np.full((h, w, c), random.uniform(180, 220), dtype=np.float32)
    haze_intensity = random.uniform(0.05, 0.15)
    img_aug = (1 - haze_intensity) * img_aug + haze_intensity * haze

    return np.clip(img_aug, 0, 255).astype(np.uint8)


def get_indian_flood_augmentation_pipeline(img_size=(512, 512), is_train=True):
    """
    Builds a composite Albumentations pipeline incorporating Indian flood domain adaptation.
    Compatible with Albumentations 2.0+.
    """
    h, w = img_size
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Lambda(image=apply_turbid_muddy_water, p=0.6),
            A.Lambda(image=apply_indian_roof_augment, p=0.4),
            A.Lambda(image=apply_monsoon_atmosphere, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        return A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
