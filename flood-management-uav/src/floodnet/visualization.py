"""Colorization and overlay helpers for segmentation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from floodnet.constants import CLASS_COLORS, CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


def palette_rgb() -> np.ndarray:
    return np.asarray(
        [CLASS_COLORS[index] for index in range(len(CLASS_NAMES))],
        dtype=np.uint8,
    )


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Return an RGB colorized mask."""
    mask = np.asarray(mask, dtype=np.int64)
    palette = palette_rgb()
    clipped = np.clip(mask, 0, len(palette) - 1)
    return palette[clipped]


def colorize_mask_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(colorize_mask(mask), cv2.COLOR_RGB2BGR)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = colorize_mask_bgr(mask)
    color = cv2.resize(color, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color, alpha, 0.0)


def denormalize_tensor(image: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to an RGB uint8 HWC array for visualization."""
    array = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array * std + mean) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def legend() -> list[dict[str, Any]]:
    return [
        {
            "id": index,
            "name": CLASS_NAMES[index],
            "color": list(CLASS_COLORS[index]),
            "is_flood": index in (1, 3, 5),
        }
        for index in range(len(CLASS_NAMES))
    ]


def side_by_side(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB image/mask comparison with the same height."""
    image_rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask_rgb = colorize_mask(mask)
    if image_rgb.shape[0] != mask_rgb.shape[0]:
        width = int(mask_rgb.shape[0] * image_rgb.shape[1] / image_rgb.shape[0])
        image_rgb = cv2.resize(image_rgb, (width, mask_rgb.shape[0]))
    return np.concatenate([image_rgb, mask_rgb], axis=1)


def save_colorized(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), colorize_mask_bgr(mask))


def save_overlay(path: Path, image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay_mask(image_bgr, mask, alpha))
