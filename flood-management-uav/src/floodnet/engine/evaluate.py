"""Evaluation entry points for classification and segmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from floodnet.config import REPO_ROOT, ensure_output_dirs, get_path, load_config, resolve_device
from floodnet.constants import NUM_CLASSES
from floodnet.data.classification_dataset import ClassificationDataset
from floodnet.data.segmentation_dataset import SegmentationDataset, one_hot_masks
from floodnet.data.transforms import (
    build_classification_transform,
    build_segmentation_transform,
)
from floodnet.engine.common import load_manifest, load_model_state, make_loader, write_json
from floodnet.losses import BCEDiceLoss
from floodnet.metrics import ClassificationMetrics, SegmentationMetrics
from floodnet.models.classifier import build_classifier
from floodnet.models.segmentor import build_segmentor
from floodnet.visualization import denormalize_tensor, save_colorized, save_overlay


def _load_task_config(args: argparse.Namespace, task: str) -> dict[str, Any]:
    default_name = "classification.yaml" if task == "classification" else "segmentation.yaml"
    config_path = args.config or (REPO_ROOT / "configs" / default_name)
    cfg = load_config(config_path)
    if args.device is not None:
        cfg["device"] = args.device
    return cfg


def _split_name(task: str, split: str, splits: dict[str, Any]) -> str:
    if split in splits:
        return split
    if split in {"test", "official_test"} and "official_test" in splits:
        return "official_test"
    if split == "official_val" and "official_val" in splits:
        return "official_val"
    available = ", ".join(sorted(splits))
    raise KeyError(f"Split {split!r} not available for {task}; available: {available}")


def _json_output(args: argparse.Namespace, cfg: dict[str, Any], filename: str) -> Path:
    if args.output is None:
        return get_path(cfg, "outputs") / args.task / filename
    output = Path(args.output)
    if output.suffix.lower() == ".json":
        return output
    return output / filename


def _predict_classification(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    model_cfg = cfg["classification"].get("model", {})
    model = build_classifier(
        arch=str(model_cfg.get("arch", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=False,
    ).to(device)
    load_model_state(args.checkpoint, model, map_location=device)
    model.eval()

    dataset = ClassificationDataset(items, transform=build_classification_transform(train=False))
    batch_size = int(args.batch_size or cfg["classification"]["train"]["batch_size"])
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="classification", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted = probabilities.argmax(dim=1).tolist()
            names = batch["name"]
            labels = batch["label"].tolist()
            for name, label, prediction, probability in zip(
                names, labels, predicted, probabilities, strict=True
            ):
                predictions.append(
                    {
                        "name": name,
                        "label": int(label),
                        "prediction": int(prediction),
                        "probabilities": [float(value) for value in probability.tolist()],
                    }
                )
    return predictions


def _evaluate_classification(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_task_config(args, "classification")
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    manifest = load_manifest(cfg, "classification")
    splits = manifest["splits"]
    split = _split_name("classification", args.split, splits)
    items = splits[split]
    if not items:
        raise ValueError(f"Classification split is empty: {split}")

    predictions = _predict_classification(cfg, args, items, device)
    labeled = [item for item in predictions if int(item["label"]) >= 0]
    metrics: dict[str, Any] | None = None
    if labeled:
        metric_state = ClassificationMetrics()
        metric_state.true = [int(item["label"]) for item in labeled]
        metric_state.predicted = [int(item["prediction"]) for item in labeled]
        metrics = metric_state.compute()

    invert = bool(args.legacy_invert) or (
        split in {"official_test", "test"}
        and bool(cfg.get("legacy", {}).get("classification_output_inversion", True))
    )
    if invert:
        for item in predictions:
            item["raw_prediction"] = item["prediction"]
            item["prediction"] = int(item["raw_prediction"]) ^ 1
            item["probabilities"] = list(reversed(item["probabilities"]))

    filename = f"eval_{split}.json" + ("_inverted" if invert else "")
    output_path = _json_output(args, cfg, filename)
    payload = {
        "task": "classification",
        "split": split,
        "checkpoint": str(args.checkpoint),
        "legacy_inverted": invert,
        "metrics": metrics,
        "num_items": len(items),
        "predictions": predictions,
    }
    write_json(output_path, payload)

    if split in {"official_test", "test"}:
        image_classes_path = output_path.parent / "image_classes.json"
        write_json(
            image_classes_path,
            {str(item["name"]): int(item["prediction"]) for item in predictions},
        )
        payload["image_classes_json"] = str(image_classes_path)
    return payload


def _evaluate_segmentation(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_task_config(args, "segmentation")
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    manifest = load_manifest(cfg, "segmentation")
    splits = manifest["splits"]
    if args.split in {"test", "official_test"}:
        raise ValueError(
            "The official test split has no masks. Use `floodnet predict` for inference instead."
        )
    split = _split_name("segmentation", args.split, splits)
    items = splits[split]
    if not items or any(not item.get("mask") for item in items):
        raise ValueError(f"Segmentation split is empty or lacks masks: {split}")

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

    image_size = manifest.get("image_size", cfg["data"]["segmentation"]["image_size"])
    legacy_val = bool(cfg["data"]["segmentation"].get("val_uses_train_transform", True))
    dataset = SegmentationDataset(
        items,
        transform=build_segmentation_transform(
            image_size,
            train=False,
            legacy_val_uses_train=legacy_val,
        ),
    )
    batch_size = int(args.batch_size or cfg["segmentation"]["train"]["batch_size"])
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
    )

    criterion = BCEDiceLoss(
        bce_weight=float(cfg["segmentation"].get("bce_weight", 0.5)),
        smooth=float(cfg["segmentation"].get("dice_smooth", 1e-5)),
    )
    metrics = SegmentationMetrics()
    total_loss = 0.0
    batches = 0
    saved: list[str] = []

    if args.output is None:
        output_dir = get_path(cfg, "outputs") / "segmentation" / f"eval_{split}"
    else:
        output_dir = Path(args.output)
        if output_dir.suffix.lower() == ".json":
            output_dir = output_dir.parent
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc="segmentation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            targets = one_hot_masks(masks, NUM_CLASSES)
            total_loss += float(criterion(logits, targets).item())
            predictions = logits.argmax(dim=1)
            metrics.update(predictions, masks)
            batches += 1

            predicted_masks = predictions.cpu().numpy().astype(np.uint8)
            for index, name in enumerate(batch["name"]):
                mask = predicted_masks[index]
                image_bgr = cv2.cvtColor(denormalize_tensor(batch["image"][index]), cv2.COLOR_RGB2BGR)
                mask_path = prediction_dir / f"{name}_mask.png"
                color_path = prediction_dir / f"{name}_color.png"
                overlay_path = prediction_dir / f"{name}_overlay.png"
                cv2.imwrite(str(mask_path), mask)
                save_colorized(color_path, mask)
                save_overlay(overlay_path, image_bgr, mask)
                saved.append(str(mask_path))

    results = metrics.compute()
    results["loss"] = total_loss / max(batches, 1)
    filename = f"eval_{split}.json"
    output_path = (
        Path(args.output)
        if args.output is not None and Path(args.output).suffix.lower() == ".json"
        else output_dir / filename
    )
    payload = {
        "task": "segmentation",
        "split": split,
        "checkpoint": str(args.checkpoint),
        "metrics": results,
        "num_items": len(items),
        "predictions_dir": str(prediction_dir),
        "prediction_files": saved,
    }
    write_json(output_path, payload)
    return payload


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.task == "classification":
        return _evaluate_classification(args)
    return _evaluate_segmentation(args)


def evaluate_cli(args: argparse.Namespace) -> int:
    evaluate(args)
    return 0
