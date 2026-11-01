"""Supervised and legacy semi-supervised DeepLabV3+ training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from floodnet.config import ensure_output_dirs, get_path, load_config, resolve_device
from floodnet.constants import NUM_CLASSES
from floodnet.data.segmentation_dataset import (
    SegmentationDataset,
    one_hot_masks,
)
from floodnet.data.transforms import build_segmentation_transform
from floodnet.engine.common import (
    CSVLogger,
    count_parameters,
    load_checkpoint,
    load_manifest,
    make_loader,
    save_checkpoint,
    set_seed,
    write_json,
)
from floodnet.losses import BCEDiceLoss
from floodnet.metrics import SegmentationMetrics
from floodnet.models.segmentor import build_segmentor
from floodnet.semi_supervised import segmentation_alpha


def _build_model(cfg: dict[str, Any], pretrained: bool) -> nn.Module:
    model_cfg = cfg["segmentation"].get("model", {})
    encoder_weights = model_cfg.get("encoder_weights", "imagenet") if pretrained else None
    return build_segmentor(
        arch=str(model_cfg.get("arch", "deeplabv3plus")),
        encoder=str(model_cfg.get("encoder", "efficientnet-b3")),
        num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
        encoder_weights=encoder_weights,
        encoder_depth=int(model_cfg.get("encoder_depth", 5)),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
    )


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict[str, Any], float]:
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            targets = one_hot_masks(masks, NUM_CLASSES)
            total_loss += float(criterion(logits, targets).item())
            metrics.update(logits.argmax(dim=1), masks)
            batches += 1
    model.train()
    results = metrics.compute()
    results["loss"] = total_loss / max(batches, 1)
    return results, float(results["miou"])


def _generate_pseudo_logits(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    store_dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Store raw logits per batch exactly as the reference semi-supervised loop does."""
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="pseudo-logits", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            predictions.append(logits.detach().to(dtype=store_dtype).cpu())
    model.train()
    return predictions


def _resolve_output_dir(cfg: dict[str, Any]) -> Path:
    output_dir = Path(str(cfg["segmentation"].get("output_dir", "checkpoints")))
    if not output_dir.is_absolute():
        output_dir = get_path(cfg, "checkpoints")
    return output_dir


def _train_supervised(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    train_cfg = cfg["segmentation"]["train"]
    manifest = load_manifest(cfg, "segmentation")
    splits = manifest["splits"]
    train_items = splits["train"]
    val_items = splits["val"]
    if not train_items or not val_items:
        raise ValueError("Segmentation manifest has empty train or validation split")

    image_size = manifest.get("image_size", cfg["data"]["segmentation"]["image_size"])
    legacy_val = bool(cfg["data"]["segmentation"].get("val_uses_train_transform", True))
    train_dataset = SegmentationDataset(
        train_items,
        transform=build_segmentation_transform(image_size, train=True),
    )
    val_dataset = SegmentationDataset(
        val_items,
        transform=build_segmentation_transform(
            image_size,
            train=False,
            legacy_val_uses_train=legacy_val,
        ),
    )
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = _build_model(cfg, pretrained=not args.no_pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    criterion = BCEDiceLoss(
        bce_weight=float(cfg["segmentation"].get("bce_weight", 0.5)),
        smooth=float(cfg["segmentation"].get("dice_smooth", 1e-5)),
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(train_cfg.get("supervised_lr", 1e-3)),
    )

    output_dir = _resolve_output_dir(cfg)
    best_path = output_dir / "seg_supervised_best.pt"
    last_path = output_dir / "seg_supervised_last.pt"
    logger = CSVLogger(
        output_dir / "segmentation_supervised_metrics.csv",
        [
            "epoch",
            "train_loss",
            "val_loss",
            "val_miou",
            "val_pixel_accuracy",
            "best_miou",
        ],
    )
    writer = (
        SummaryWriter(
            log_dir=str(get_path(cfg, "outputs") / "tensorboard" / "segmentation_supervised")
        )
        if bool(cfg.get("logging", {}).get("tensorboard", True))
        else None
    )

    epochs = int(train_cfg.get("supervised_epochs", 100))
    if args.smoke:
        epochs = 1
    best_miou = -1.0
    for epoch in range(epochs):
        started = time.time()
        epoch_label = epoch + 1
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in tqdm(
            train_loader, desc=f"supervised {epoch_label}/{epochs}", leave=False
        ):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            targets = one_hot_masks(masks, NUM_CLASSES)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        train_loss = loss_sum / max(batches, 1)
        val_metrics, val_miou = _evaluate(model, val_loader, device, criterion)
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch_label,
                cfg,
                val_metrics,
                extra={"task": "segmentation", "stage": "supervised", "best_miou": best_miou},
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch_label,
            cfg,
            val_metrics,
            extra={"task": "segmentation", "stage": "supervised", "best_miou": best_miou},
        )

        row = {
            "epoch": epoch_label,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_miou,
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "best_miou": best_miou,
        }
        logger.log(row)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(f"segmentation_supervised/{key}", value, epoch_label)
        elapsed = time.time() - started
        print(
            f"[supervised] epoch {epoch_label}/{epochs} | loss={train_loss:.4f} | "
            f"val_miou={val_miou:.4f} | best={best_miou:.4f} | {elapsed:.1f}s"
        )

    if writer is not None:
        writer.close()
    write_json(
        get_path(cfg, "outputs") / "segmentation" / "supervised_summary.json",
        {"best_miou": best_miou, "epochs": epochs, "best_checkpoint": str(best_path)},
    )
    return best_path


def _train_semi_supervised(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    resume_path: Path,
) -> Path:
    train_cfg = cfg["segmentation"]["train"]
    manifest = load_manifest(cfg, "segmentation")
    splits = manifest["splits"]
    train_items = splits["train"]
    val_items = splits["val"]
    unlabeled_items = splits.get("unlabeled", [])
    if not unlabeled_items:
        raise ValueError("Segmentation manifest has no unlabeled split")

    image_size = manifest.get("image_size", cfg["data"]["segmentation"]["image_size"])
    legacy_val = bool(cfg["data"]["segmentation"].get("val_uses_train_transform", True))
    train_dataset = SegmentationDataset(
        train_items,
        transform=build_segmentation_transform(image_size, train=True),
    )
    val_dataset = SegmentationDataset(
        val_items,
        transform=build_segmentation_transform(
            image_size,
            train=False,
            legacy_val_uses_train=legacy_val,
        ),
    )
    unlabeled_dataset = SegmentationDataset(
        unlabeled_items,
        transform=build_segmentation_transform(image_size, train=True),
        unlabeled=True,
    )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    unlabeled_loader = make_loader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = _build_model(cfg, pretrained=not args.no_pretrained).to(device)
    print(f"Resuming semi-supervised stage from {resume_path}")
    checkpoint = load_checkpoint(resume_path, model, optimizer=None, map_location=device)
    print(f"Loaded checkpoint epoch={checkpoint.get('epoch')} metrics={checkpoint.get('metrics')}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = BCEDiceLoss(
        bce_weight=float(cfg["segmentation"].get("bce_weight", 0.5)),
        smooth=float(cfg["segmentation"].get("dice_smooth", 1e-5)),
    )
    optimizer = SGD(model.parameters(), lr=float(train_cfg.get("semi_supervised_lr", 0.01)))

    output_dir = _resolve_output_dir(cfg)
    best_path = output_dir / str(cfg["segmentation"].get("checkpoint_name", "seg_best.pt"))
    last_path = output_dir / "seg_last.pt"
    logger = CSVLogger(
        output_dir / "segmentation_semi_supervised_metrics.csv",
        [
            "epoch",
            "alpha",
            "train_loss",
            "labeled_loss",
            "unlabeled_loss",
            "val_loss",
            "val_miou",
            "val_pixel_accuracy",
            "best_miou",
        ],
    )
    writer = (
        SummaryWriter(
            log_dir=str(get_path(cfg, "outputs") / "tensorboard" / "segmentation_semi_supervised")
        )
        if bool(cfg.get("logging", {}).get("tensorboard", True))
        else None
    )

    epochs = int(train_cfg.get("semi_supervised_epochs", 100))
    if args.smoke:
        epochs = 1
    alpha_start = int(train_cfg.get("alpha_start", 15))
    alpha_end = int(train_cfg.get("alpha_end", 655))
    alpha_max = float(train_cfg.get("alpha_max", 0.5))
    pseudo_every = 14
    save_every = int(train_cfg.get("save_every", 5))
    store_dtype = (
        torch.float16
        if str(cfg.get("legacy", {}).get("segmentation_pseudo_store_dtype", "float32"))
        == "float16"
        else torch.float32
    )
    pseudo_logits: list[torch.Tensor] | None = None
    best_miou = -1.0

    for epoch in range(epochs):
        started = time.time()
        epoch_label = epoch + 1
        alpha = segmentation_alpha(epoch, alpha_start, alpha_end, alpha_max)

        model.train()
        labeled_loss_sum = 0.0
        unlabeled_loss_sum = 0.0
        labeled_batches = 0
        unlabeled_batches = 0
        for batch in tqdm(
            train_loader, desc=f"semi {epoch_label}/{epochs} labeled", leave=False
        ):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            targets = one_hot_masks(masks, NUM_CLASSES)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            labeled_loss_sum += float(loss.item())
            labeled_batches += 1

        if epoch >= pseudo_every and unlabeled_items:
            pseudo_logits = _generate_pseudo_logits(
                model, unlabeled_loader, device, store_dtype
            )
            print(
                f"Regenerated pseudo logits for {len(unlabeled_items)} images "
                f"({len(pseudo_logits)} batches, dtype={store_dtype})"
            )

        if alpha > 0.0 and pseudo_logits is not None:
            model.train()
            for batch_index, batch in enumerate(
                tqdm(
                    unlabeled_loader,
                    desc=f"semi {epoch_label}/{epochs} unlabeled",
                    leave=False,
                )
            ):
                if batch_index >= len(pseudo_logits):
                    break
                images = batch["image"].to(device, non_blocking=True)
                targets = pseudo_logits[batch_index].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = alpha * criterion(logits, targets)
                loss.backward()
                optimizer.step()
                unlabeled_loss_sum += float(loss.item())
                unlabeled_batches += 1

        labeled_loss = labeled_loss_sum / max(labeled_batches, 1)
        unlabeled_loss = unlabeled_loss_sum / max(unlabeled_batches, 1)
        train_loss = labeled_loss + unlabeled_loss
        val_metrics, val_miou = _evaluate(model, val_loader, device, criterion)

        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch_label,
                cfg,
                val_metrics,
                extra={
                    "task": "segmentation",
                    "stage": "semi-supervised",
                    "best_miou": best_miou,
                    "alpha": alpha,
                },
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch_label,
            cfg,
            val_metrics,
            extra={
                "task": "segmentation",
                "stage": "semi-supervised",
                "best_miou": best_miou,
                "alpha": alpha,
            },
        )
        if save_every > 0 and epoch_label % save_every == 0:
            save_checkpoint(
                output_dir / f"seg_semi_epoch_{epoch_label:03d}.pt",
                model,
                optimizer,
                epoch_label,
                cfg,
                val_metrics,
                extra={
                    "task": "segmentation",
                    "stage": "semi-supervised",
                    "best_miou": best_miou,
                    "alpha": alpha,
                },
            )

        row = {
            "epoch": epoch_label,
            "alpha": alpha,
            "train_loss": train_loss,
            "labeled_loss": labeled_loss,
            "unlabeled_loss": unlabeled_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_miou,
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "best_miou": best_miou,
        }
        logger.log(row)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(f"segmentation_semi_supervised/{key}", value, epoch_label)
        elapsed = time.time() - started
        print(
            f"[semi] epoch {epoch_label}/{epochs} | alpha={alpha:.5f} | "
            f"loss={train_loss:.4f} | val_miou={val_miou:.4f} | "
            f"best={best_miou:.4f} | {elapsed:.1f}s"
        )

    if writer is not None:
        writer.close()
    write_json(
        get_path(cfg, "outputs") / "segmentation" / "semi_supervised_summary.json",
        {"best_miou": best_miou, "epochs": epochs, "best_checkpoint": str(best_path)},
    )
    return best_path


def train_segmentor(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    train_cfg = cfg["segmentation"]["train"]
    if args.supervised_epochs is not None:
        train_cfg["supervised_epochs"] = int(args.supervised_epochs)
    if args.semi_supervised_epochs is not None:
        train_cfg["semi_supervised_epochs"] = int(args.semi_supervised_epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        cfg["device"] = args.device
    if args.smoke:
        train_cfg["supervised_epochs"] = 1
        train_cfg["semi_supervised_epochs"] = 1
        train_cfg["batch_size"] = min(int(train_cfg["batch_size"]), 2)
        train_cfg["num_workers"] = 0
        train_cfg["save_every"] = 0
    if args.no_pretrained:
        cfg["segmentation"].setdefault("model", {})["encoder_weights"] = None

    seed = int(cfg["seed"])
    set_seed(seed)
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    output_dir = _resolve_output_dir(cfg)
    if args.resume is not None:
        resume_path = args.resume
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        resume_path = output_dir / "seg_supervised_best.pt"

    final_path: Path | None = None
    if args.stage in {"supervised", "both"} and not (
        args.stage == "both" and args.resume is not None
    ):
        final_path = _train_supervised(cfg, args, device)
        if args.stage == "supervised":
            return final_path

    if args.stage in {"semi-supervised", "both"}:
        if not resume_path.exists():
            raise FileNotFoundError(
                f"No supervised checkpoint found at {resume_path}. "
                "Run `--stage supervised` first or pass `--resume`."
            )
        final_path = _train_semi_supervised(cfg, args, device, resume_path)
    if final_path is None:
        raise ValueError(f"Unsupported stage: {args.stage}")
    return final_path


def train_segmentor_cli(args: argparse.Namespace) -> int:
    train_segmentor(args)
    return 0
