"""
Training Script for UAV Flood Mapping Semantic Segmentation Model (DeepLabV3+).
Features:
- Focal + Dice combined loss for extreme class imbalance.
- Mixed precision training with torch.amp.
- Comprehensive metric tracking: Mean IoU (mIoU), F1-Score, Class-specific IoU.
- Best model checkpointing based on Validation mIoU.
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.segmentor import DeepLabV3Plus, FocalDiceLoss
from data.dataset import FloodNetSegmentationDataset


def compute_iou(confusion_matrix):
    """
    Computes per-class IoU and mean IoU from a confusion matrix.
    confusion_matrix: [num_classes, num_classes] (rows: true, cols: pred)
    """
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    prediction_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + prediction_set - intersection

    iou = np.zeros(len(intersection))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    miou = np.mean(iou[valid]) if np.any(valid) else 0.0

    return iou, miou


def update_confusion_matrix(matrix, pred, target, num_classes):
    """Updates confusion matrix in-place."""
    mask = (target >= 0) & (target < num_classes)
    p = pred[mask].astype(int)
    t = target[mask].astype(int)
    bins = t * num_classes + p
    counts = np.bincount(bins, minlength=num_classes * num_classes)
    matrix += counts.reshape((num_classes, num_classes))


def train_epoch(model, loader, optimizer, criterion, scaler, device, num_classes):
    model.train()
    running_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        for p, t in zip(preds, targets_np):
            update_confusion_matrix(conf_matrix, p, t, num_classes)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    _, miou = compute_iou(conf_matrix)
    return epoch_loss, miou


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            for p, t in zip(preds, targets_np):
                update_confusion_matrix(conf_matrix, p, t, num_classes)

    val_loss = running_loss / len(loader.dataset)
    class_ious, miou = compute_iou(conf_matrix)
    accuracy = np.diag(conf_matrix).sum() / max(conf_matrix.sum(), 1)

    return val_loss, miou, class_ious, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train UAV FloodNet Segmentation Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    seg_cfg = cfg["segmentor"]
    epochs = args.epochs or seg_cfg["epochs"]
    batch_size = args.batch_size or seg_cfg["batch_size"]
    lr = args.lr or seg_cfg["lr"]
    num_classes = seg_cfg["num_classes"]
    img_size = tuple(seg_cfg["img_size"])

    # Setup directories
    checkpoint_dir = cfg["paths"]["models_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Dataset
    data_root = cfg["paths"]["data_root"]
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
        print(f"Dataset not found at {img_dir}. Generating sample dataset...")
        from data.download_floodnet import create_verification_samples
        create_verification_samples(base_dir=data_root, num_samples=24)

    full_dataset = FloodNetSegmentationDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        is_binary=False,
        apply_indian_adaptation=True,
        img_size=img_size,
        is_train=True,
    )

    # Train / Val Split (80% / 20%)
    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["project"]["seed"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Loaded {train_size} training samples and {val_size} validation samples.")

    # Initialize Model, Loss, Optimizer
    model = DeepLabV3Plus(num_classes=num_classes, pretrained=True).to(device)
    criterion = FocalDiceLoss(num_classes=num_classes, gamma=2.0, alpha=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=seg_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler()

    best_miou = 0.0
    history = {"train_loss": [], "train_miou": [], "val_loss": [], "val_miou": [], "val_acc": []}

    print("\nStarting DeepLabV3+ Training for Flood Mapping...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} (lr: {optimizer.param_groups[0]['lr']:.6f}) ---")
        train_loss, train_miou = train_epoch(model, train_loader, optimizer, criterion, scaler, device, num_classes)
        val_loss, val_miou, class_ious, val_acc = validate(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val mIoU:   {val_miou:.4f} | Val Acc: {val_acc:.4f}")

        # Class specific details for critical flood classes
        class_names = FloodNetSegmentationDataset.CLASS_NAMES
        print(f"  Water IoU: {class_ious[5]:.4f} | Flooded Road IoU: {class_ious[3]:.4f} | Flooded Building IoU: {class_ious[1]:.4f}")

        # Save checkpoint
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
            best_path = os.path.join(checkpoint_dir, "best_segmentor.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "class_ious": class_ious.tolist(),
            }, best_path)
            print(f"  >> Best model saved to {best_path} (mIoU: {best_miou:.4f})")

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_segmentor.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_miou,
        }, latest_path)

    # Save training history
    history_path = os.path.join(checkpoint_dir, "segmentor_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History written to {history_path}")


if __name__ == "__main__":
    main()
