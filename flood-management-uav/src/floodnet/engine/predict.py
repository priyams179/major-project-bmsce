"""Inference for classification and segmentation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from floodnet.config import REPO_ROOT, ensure_output_dirs, load_config, resolve_device
from floodnet.constants import (
    CLASS_NAMES,
    FLOOD_CLASSES,
    NUM_CLASSES,
    SEGMENTATION_OUTPUT_SIZE,
)
from floodnet.data.transforms import (
    build_classification_transform,
    build_segmentation_transform,
)
from floodnet.engine.common import load_model_state, write_json
from floodnet.models.classifier import build_classifier
from floodnet.models.segmentor import build_segmentor
from floodnet.visualization import save_colorized, save_overlay


def _load_task_config(args: argparse.Namespace, task: str) -> dict[str, Any]:
    default_name = "classification.yaml" if task == "classification" else "segmentation.yaml"
    config_path = args.config or (REPO_ROOT / "configs" / default_name)
    cfg = load_config(config_path)
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def _collect_images(input_path: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    images = sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )
    if not images:
        raise ValueError(f"No images found in {input_path}")
    return images


def _predict_classification_image(
    model: torch.nn.Module,
    image_path: Path,
    image_size: list[int],
    transform: Any,
    device: torch.device,
) -> tuple[int, list[float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = int(image_size[0]), int(image_size[1])
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    return int(np.argmax(probabilities)), [float(value) for value in probabilities]


def _predict_segmentation_image(
    model: torch.nn.Module,
    image_path: Path,
    image_size: list[int],
    transform: Any,
    device: torch.device,
    output_size: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    original_h, original_w = image_bgr.shape[:2]
    height, width = int(image_size[0]), int(image_size[1])
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((2, 2), np.uint8)
    preprocessed = cv2.bilateralFilter(resized, 5, 75, 75)
    preprocessed = cv2.dilate(preprocessed, kernel, iterations=2)
    preprocessed = cv2.erode(preprocessed, kernel, iterations=1)
    rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    augmented = transform(image=rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()
    mask = probabilities.argmax(axis=0).astype(np.uint8)

    if output_size is None:
        target_w, target_h = SEGMENTATION_OUTPUT_SIZE
    else:
        target_w, target_h = output_size
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    erosion_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, erosion_kernel, iterations=1)
    return mask, image_bgr, probabilities


def _parse_output_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"output-size must be WxH, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"output-size must be positive, got {value!r}")
    return width, height


def _count_classes(mask: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    total = int(counts.sum())
    pixels = {CLASS_NAMES[index]: int(counts[index]) for index in range(NUM_CLASSES)}
    fractions = {
        CLASS_NAMES[index]: float(counts[index] / total) if total else 0.0
        for index in range(NUM_CLASSES)
    }
    flood_pixels = int(sum(int(counts[index]) for index in FLOOD_CLASSES))
    return {
        "pixels": pixels,
        "fractions": fractions,
        "flood_pixels": flood_pixels,
        "flood_fraction": float(flood_pixels / total) if total else 0.0,
    }


def _predict_classification(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_task_config(args, "classification")
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    model_cfg = cfg["classification"].get("model", {})
    model = build_classifier(
        arch=str(model_cfg.get("arch", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=False,
    ).to(device)
    load_model_state(args.checkpoint, model, map_location=device)
    model.eval()

    image_size = cfg["data"]["classification"]["image_size"]
    transform = build_classification_transform(train=False)
    image_paths = _collect_images(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.legacy_invert is None:
        invert = bool(cfg.get("legacy", {}).get("classification_output_inversion", True))
    else:
        invert = bool(args.legacy_invert)
    predictions: list[dict[str, Any]] = []
    image_classes: dict[str, int] = {}
    for image_path in image_paths:
        raw_prediction, probabilities = _predict_classification_image(
            model, image_path, image_size, transform, device
        )
        final_prediction = raw_prediction ^ 1 if invert else raw_prediction
        image_classes[image_path.stem] = final_prediction
        predictions.append(
            {
                "name": image_path.stem,
                "path": str(image_path),
                "raw_prediction": raw_prediction,
                "prediction": final_prediction,
                "label": "Flooded" if final_prediction == 1 else "Non-Flooded",
                "probabilities": probabilities,
            }
        )

    payload = {
        "task": "classification",
        "checkpoint": str(args.checkpoint),
        "legacy_inverted": invert,
        "predictions": predictions,
    }
    write_json(output_dir / "predictions.json", payload)
    write_json(output_dir / "image_classes.json", image_classes)
    return payload


def _predict_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_task_config(args, "segmentation")
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    model_cfg = cfg["segmentation"].get("model", {})
    model = build_segmentor(
        arch=str(model_cfg.get("arch", "deeplabv3plus")),
        encoder=str(model_cfg.get("encoder", "efficientnet-b3")),
        num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
        encoder_weights=None,
        encoder_depth=int(model_cfg.get("encoder_depth", 5)),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
    ).to(device)
    load_model_state(args.checkpoint, model, map_location=device)
    model.eval()

    image_size = cfg["data"]["segmentation"]["image_size"]
    transform = build_segmentation_transform(image_size, train=False, legacy_val_uses_train=False)
    output_size = _parse_output_size(args.output_size)
    image_paths = _collect_images(args.input)
    output_dir = Path(args.output)
    mask_dir = output_dir / "masks"
    color_dir = output_dir / "color"
    overlay_dir = output_dir / "overlays"
    for directory in (mask_dir, color_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for image_path in image_paths:
        mask, image_bgr, _ = _predict_segmentation_image(
            model, image_path, image_size, transform, device, output_size
        )
        mask_path = mask_dir / f"{image_path.stem}.png"
        color_path = color_dir / f"{image_path.stem}.png"
        overlay_path = overlay_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        save_colorized(color_path, mask)
        overlay_image = cv2.resize(
            image_bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA
        )
        save_overlay(overlay_path, overlay_image, mask)
        summary = {
            "name": image_path.stem,
            "path": str(image_path),
            "mask": str(mask_path),
            "color": str(color_path),
            "overlay": str(overlay_path),
            "shape": list(mask.shape),
        }
        summary.update(_count_classes(mask))
        summaries.append(summary)

    payload = {
        "task": "segmentation",
        "checkpoint": str(args.checkpoint),
        "output_size": list(output_size) if output_size else list(SEGMENTATION_OUTPUT_SIZE),
        "predictions": summaries,
    }
    write_json(output_dir / "predictions.json", payload)
    return payload


def predict(args: argparse.Namespace) -> dict[str, Any]:
    if args.task == "classification":
        return _predict_classification(args)
    return _predict_segmentation(args)


def predict_cli(args: argparse.Namespace) -> int:
    predict(args)
    return 0
