"""Semi-supervised ResNet18 flood classifier matching FloodNet_T1.ipynb."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from floodnet.config import ensure_output_dirs, get_path, load_config, resolve_device
from floodnet.data.classification_dataset import ClassificationDataset
from floodnet.data.transforms import build_classification_transform
from floodnet.engine.common import (
    CSVLogger,
    count_parameters,
    load_manifest,
    make_loader,
    save_checkpoint,
    set_seed,
    write_json,
)
from floodnet.metrics import ClassificationMetrics
from floodnet.models.classifier import build_classifier
from floodnet.semi_supervised import classification_alpha


def _weighted_sampler(items: list[dict[str, Any]]) -> WeightedRandomSampler:
    labels = np.asarray([int(item["label"]) for item in items])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_weights = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, Any], float]:
    model.eval()
    metrics = ClassificationMetrics()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="validate", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(loss_fn(logits, labels).item())
            metrics.update(logits, labels)
            batches += 1
    model.train()
    results = metrics.compute()
    results["loss"] = total_loss / max(batches, 1)
    return results, float(results["macro_f1"])


def _generate_pseudo_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    size: int,
) -> torch.Tensor:
    model.eval()
    pseudo = torch.zeros(size, dtype=torch.long)
    with torch.no_grad():
        for batch in tqdm(loader, desc="pseudo-labels", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1).reshape(-1).cpu()
            indices = batch["index"].reshape(-1).cpu()
            pseudo[indices] = predictions
    model.train()
    return pseudo


def train_classifier(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    train_cfg = cfg["classification"]["train"]
    model_cfg = cfg["classification"].get("model", {})
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        cfg["device"] = args.device
    if args.smoke:
        train_cfg["epochs"] = 1
        train_cfg["batch_size"] = min(int(train_cfg["batch_size"]), 4)
        train_cfg["num_workers"] = 0
    if args.no_pretrained:
        model_cfg["pretrained"] = False

    seed = int(cfg["seed"])
    set_seed(seed)
    ensure_output_dirs(cfg)
    device = torch.device(resolve_device(cfg))
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    manifest = load_manifest(cfg, "classification")
    splits = manifest["splits"]
    train_items = splits["train"]
    val_items = splits["val"]
    unlabeled_items = splits.get("unlabeled", [])
    if not train_items or not val_items:
        raise ValueError("Classification manifest has empty train or validation split")
    print(
        f"Classification split sizes: train={len(train_items)}, "
        f"val={len(val_items)}, unlabeled={len(unlabeled_items)}"
    )

    train_transform = build_classification_transform(train=True)
    val_transform = build_classification_transform(train=False)
    train_dataset = ClassificationDataset(train_items, transform=train_transform)
    val_dataset = ClassificationDataset(val_items, transform=val_transform)
    unlabeled_dataset = ClassificationDataset(unlabeled_items, transform=val_transform)

    sampler = _weighted_sampler(train_items)
    num_workers = int(train_cfg.get("num_workers", 4))
    batch_size = int(train_cfg["batch_size"])
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
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
        shuffle=True,
        num_workers=num_workers,
    )

    model = build_classifier(
        arch=str(model_cfg.get("arch", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
    ).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.01)),
        momentum=float(train_cfg.get("momentum", 0.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    output_dir = Path(str(cfg["classification"].get("output_dir", "checkpoints")))
    if not output_dir.is_absolute():
        output_dir = get_path(cfg, "checkpoints")
    best_path = output_dir / str(cfg["classification"].get("checkpoint_name", "cls_best.pt"))
    last_path = output_dir / "cls_last.pt"
    logger = CSVLogger(
        output_dir / "classification_metrics.csv",
        [
            "epoch",
            "alpha",
            "train_loss",
            "labeled_loss",
            "unlabeled_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "best_macro_f1",
        ],
    )
    writer = (
        SummaryWriter(log_dir=str(get_path(cfg, "outputs") / "tensorboard" / "classification"))
        if bool(cfg.get("logging", {}).get("tensorboard", True))
        else None
    )

    epochs = int(train_cfg["epochs"])
    alpha_start = int(train_cfg.get("alpha_start", 10))
    alpha_end = int(train_cfg.get("alpha_end", 150))
    alpha_max = float(train_cfg.get("alpha_max", 1.0))
    pseudo_every = 9
    best_f1 = -1.0
    pseudo_labels: torch.Tensor | None = None

    for epoch in range(epochs):
        started = time.time()
        epoch_label = epoch + 1
        alpha = classification_alpha(epoch, alpha_start, alpha_end, alpha_max)

        model.train()
        labeled_loss_sum = 0.0
        unlabeled_loss_sum = 0.0
        labeled_batches = 0
        unlabeled_batches = 0

        for batch in tqdm(
            train_loader, desc=f"epoch {epoch_label}/{epochs} labeled", leave=False
        ):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            labeled_loss_sum += float(loss.item())
            labeled_batches += 1

        if epoch >= pseudo_every and unlabeled_items:
            pseudo_labels = _generate_pseudo_labels(
                model, unlabeled_loader, device, len(unlabeled_items)
            )
            print(f"Regenerated pseudo labels for {len(unlabeled_items)} images")

        if alpha > 0.0 and pseudo_labels is not None and unlabeled_items:
            model.train()
            for batch in tqdm(
                unlabeled_loader,
                desc=f"epoch {epoch_label}/{epochs} unlabeled",
                leave=False,
            ):
                images = batch["image"].to(device, non_blocking=True)
                indices = batch["index"].cpu()
                targets = pseudo_labels[indices].to(device, non_blocking=True)
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
        val_metrics, val_f1 = _evaluate(model, val_loader, device, criterion)

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch_label,
                cfg,
                val_metrics,
                extra={"task": "classification", "best_macro_f1": best_f1},
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch_label,
            cfg,
            val_metrics,
            extra={"task": "classification", "best_macro_f1": best_f1},
        )

        row = {
            "epoch": epoch_label,
            "alpha": alpha,
            "train_loss": train_loss,
            "labeled_loss": labeled_loss,
            "unlabeled_loss": unlabeled_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_f1,
            "best_macro_f1": best_f1,
        }
        logger.log(row)
        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(f"classification/{key}", value, epoch_label)
        elapsed = time.time() - started
        print(
            f"Epoch {epoch_label}/{epochs} | alpha={alpha:.4f} | loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_f1:.4f} | "
            f"best_f1={best_f1:.4f} | {elapsed:.1f}s"
        )

    write_json(
        get_path(cfg, "outputs") / "classification" / "training_summary.json",
        {"best_macro_f1": best_f1, "epochs": epochs, "best_checkpoint": str(best_path)},
    )
    if writer is not None:
        writer.close()
    return best_path


def train_classifier_cli(args: argparse.Namespace) -> int:
    train_classifier(args)
    return 0
