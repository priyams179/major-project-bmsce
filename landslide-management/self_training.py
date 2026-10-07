#!/usr/bin/env python3
"""
Milestone 2: Self-training with pseudo-labels (the lever shared by ALL four
winning solutions of the Landslide4Sense 2022 competition).

The L4S training sites differ from the val/test sites, so a model trained
only on `train` faces a documented distribution shift.  The winning teams
bridged it by pseudo-labelling the *unlabelled* val/test patches with a
teacher model and retraining including them:

  1. Teacher = any trained checkpoint (e.g. the Milestone-1 winner).
  2. Teacher predicts probability maps on the val and test patches.
  3. Confident pixels only become pseudo-labels:
       p >= pos-thresh  -> pseudo-label 1
       p <= neg-thresh  -> pseudo-label 0
       in between       -> excluded (validity 0, ignored by the masked loss)
  4. A fresh student is trained on real train labels + pseudo-labelled
     val/test images, validated on the REAL val labels each epoch.
  5. Best-val student checkpoint is evaluated once on the REAL test labels.

NOTE: this is transductive — the student has seen the test *images*
(unlabelled) during training, exactly like the competition's test-phase
solutions.  Report the number in those terms.

Usage:
    python self_training.py --teacher models/m1_sepnorm_lov/best_model.pth \
        --epochs 30 --output-dir models/m2_selftrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from models import get_model, MaskedBCELovaszLoss, MaskedBCEDiceLoss, \
    MaskedLovaszHingeLoss
from train_segmentation import LandslideDataset, evaluate, \
    compute_or_load_split_stats


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------
def generate_pseudo_labels(teacher_paths: list[str], args, stats: dict) -> None:
    """Run an ensemble of teacher models with TTA over val+test and write confident pseudo-masks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers = []
    for path in teacher_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = get_model(
            arch=ckpt["arch"],
            in_channels=ckpt["in_channels"],
            base_features=ckpt.get("base_features", 32),
            decoder_dropout=ckpt.get("decoder_dropout", 0.0),
        ).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        t_norm = ckpt.get("norm", "none")
        t_scale = ckpt.get("scale", 1)
        teachers.append((m, ckpt, t_norm, t_scale))
        print(f"Teacher: {ckpt['arch']} (scale x{t_scale}, val IoU "
              f"{ckpt.get('val_iou', float('nan')) * 100:.2f}%) from {path}")

    target_scale = args.scale
    target_hw = 128 * target_scale

    for split in ("val", "test"):
        out_dir = os.path.join(args.pseudo_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        base_ds = LandslideDataset(
            args.data_dir, split=split, augment=False,
            norm="none", scale=1, stats=None
        )
        n_samples = len(base_ds)
        prob_sum = [None] * n_samples

        for m, ckpt, t_norm, t_scale in teachers:
            t_stats = compute_or_load_split_stats(args.data_dir) if t_norm != "none" else None
            ds = LandslideDataset(
                args.data_dir, split=split, augment=False,
                norm=t_norm, scale=t_scale, stats=t_stats
            )
            with torch.no_grad():
                for i in range(n_samples):
                    img_t, _, _ = ds[i]
                    x = img_t.unsqueeze(0).to(device)
                    # 4-fold TTA
                    x_h = torch.flip(x, dims=[-1])
                    x_v = torch.flip(x, dims=[-2])
                    x_hv = torch.flip(x, dims=[-2, -1])
                    p0 = torch.sigmoid(m(x))
                    ph = torch.flip(torch.sigmoid(m(x_h)), dims=[-1])
                    pv = torch.flip(torch.sigmoid(m(x_v)), dims=[-2])
                    phv = torch.flip(torch.sigmoid(m(x_hv)), dims=[-2, -1])
                    p = ((p0 + ph + pv + phv) / 4.0)[0, 0]
                    if p.shape[-1] != target_hw:
                        p = F.interpolate(p.unsqueeze(0).unsqueeze(0),
                                          size=(target_hw, target_hw),
                                          mode="bilinear", align_corners=False)[0, 0]
                    p_np = p.cpu().numpy()
                    if prob_sum[i] is None:
                        prob_sum[i] = p_np.astype(np.float32)
                    else:
                        prob_sum[i] += p_np.astype(np.float32)

        n_confident = 0
        n_total = 0
        for i in range(n_samples):
            prob = prob_sum[i] / len(teachers)
            pseudo = np.where(prob >= args.pos_thresh, 1.0,
                              np.where(prob <= args.neg_thresh, 0.0,
                                       -1.0)).astype(np.float32)
            conf = (pseudo >= 0).astype(np.float32)             # validity
            pseudo = np.clip(pseudo, 0.0, 1.0) * conf           # label

            np.save(os.path.join(out_dir, f"pseudo_mask_{i:05d}.npy"), pseudo)
            np.save(os.path.join(out_dir, f"pseudo_valid_{i:05d}.npy"), conf)
            n_confident += int(conf.sum())
            n_total += conf.size

        pct = 100.0 * n_confident / max(n_total, 1)
        print(f"  {split:4s}: {n_samples} patches -> {pct:.1f}% pixels "
              f"confident [{args.neg_thresh}, {args.pos_thresh}] (resolution {target_hw}x{target_hw})")

    for m, _, _, _ in teachers:
        del m
    torch.cuda.empty_cache()


class PseudoDataset(Dataset):
    """Val/test patches paired with teacher pseudo-labels (no augmentation)."""

    def __init__(self, data_dir: str, pseudo_dir: str, split: str,
                 norm: str, scale: int, stats: dict):
        self.img_ds = LandslideDataset(
            data_dir, split=split, augment=False,
            norm=norm, scale=scale, stats=stats,
        )
        self.pseudo_dir = os.path.join(pseudo_dir, split)

    def __len__(self) -> int:
        return len(self.img_ds)

    def __getitem__(self, idx: int):
        img_t, _, _ = self.img_ds[idx]
        pseudo = np.load(os.path.join(
            self.pseudo_dir, f"pseudo_mask_{idx:05d}.npy"))
        conf = np.load(os.path.join(
            self.pseudo_dir, f"pseudo_valid_{idx:05d}.npy"))
        return (img_t,
                torch.from_numpy(pseudo).unsqueeze(0),
                torch.from_numpy(conf).unsqueeze(0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--teacher", nargs="+", required=True,
                   help="Teacher checkpoint(s) (.pth with full config metadata)")
    p.add_argument("--data-dir", default="data/processed/l4s")
    p.add_argument("--pseudo-dir", default="data/processed/l4s_pseudo")
    p.add_argument("--norm", choices=["none", "train", "split"], default="split",
                   help="Must match the teacher's preprocessing (default split)")
    p.add_argument("--scale", type=int, default=1,
                   help="Must match the teacher's scale promotion")
    p.add_argument("--arch", default=None,
                   help="Student arch (default: first teacher's arch)")
    p.add_argument("--loss", choices=["bce_dice", "bce_lovasz", "lovasz"],
                   default="bce_lovasz")
    p.add_argument("--pos-thresh", type=float, default=0.7,
                   help="P>=this becomes pseudo-positive (1st place used 0.7)")
    p.add_argument("--neg-thresh", type=float, default=0.3,
                   help="P<=this becomes pseudo-negative; between = ignored")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", default="models/m2_selftrain")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    use_amp = args.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    teacher_paths = args.teacher
    first_teacher_ckpt = torch.load(teacher_paths[0], map_location="cpu",
                                    weights_only=False)
    student_arch = args.arch or first_teacher_ckpt["arch"]
    student_in_channels = first_teacher_ckpt.get("in_channels", 14)
    stats = compute_or_load_split_stats(args.data_dir) \
        if args.norm != "none" else None

    # --- Step 1+2: pseudo-labels from the teacher ensemble ------------------
    if not os.path.isdir(os.path.join(args.pseudo_dir, "test")):
        print("=" * 70)
        print("Generating pseudo-labels with teacher ensemble (with TTA)")
        print("=" * 70)
        generate_pseudo_labels(teacher_paths, args, stats)
    else:
        print(f"Reusing existing pseudo-labels in {args.pseudo_dir}")

    # --- Step 3: datasets ---------------------------------------------------
    print("\nBuilding training set: real train + pseudo val + pseudo test")
    real_train = LandslideDataset(
        args.data_dir, split="train", augment=True,
        norm=args.norm, scale=args.scale, stats=stats,
    )
    pseudo_val = PseudoDataset(args.data_dir, args.pseudo_dir, "val",
                               args.norm, args.scale, stats)
    pseudo_test = PseudoDataset(args.data_dir, args.pseudo_dir, "test",
                                args.norm, args.scale, stats)
    train_set = ConcatDataset([real_train, pseudo_val, pseudo_test])
    # Validation ALWAYS uses the real val labels (selection discipline).
    val_set = LandslideDataset(
        args.data_dir, split="val", augment=False,
        norm=args.norm, scale=args.scale, stats=stats,
    )
    test_set = LandslideDataset(
        args.data_dir, split="test", augment=False,
        norm=args.norm, scale=args.scale, stats=stats,
    )
    print(f"  train {len(real_train)} real + {len(pseudo_val)} + "
          f"{len(pseudo_test)} pseudo = {len(train_set)} patches")

    pin = device.type == "cuda"
    loader_args = dict(num_workers=args.workers, pin_memory=pin,
                       persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, **loader_args)

    # --- Step 4: student training ------------------------------------------
    model = get_model(arch=student_arch,
                      in_channels=student_in_channels).to(device)
    if args.loss == "bce_dice":
        criterion = MaskedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    elif args.loss == "lovasz":
        criterion = MaskedLovaszHingeLoss()
    else:
        criterion = MaskedBCELovaszLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\nStudent: {student_arch} | loss {args.loss} | norm {args.norm} "
          f"| scale x{args.scale} | AMP {use_amp}")
    header = (f"{'Epoch':>6} | {'Trn Loss':>9} | {'Val IoU':>8} | "
              f"{'Val F1':>8} | {'Prec':>7} | {'Recall':>7} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    best_val_iou = -1.0
    history = {"train_loss": [], "val_iou": [], "val_f1": []}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss = 0.0
        nb = 0
        for x, y, m in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss = criterion(model(x), y, mask=m)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y, mask=m)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            run_loss += loss.item()
            nb += 1
        scheduler.step()
        train_loss = run_loss / max(nb, 1)

        _, vm = evaluate(model, val_loader, criterion, device, scaler)
        history["train_loss"].append(train_loss)
        history["val_iou"].append(vm["iou"])
        history["val_f1"].append(vm["f1"])
        print(f"{epoch:>6} | {train_loss:>9.4f} | {vm['iou']*100:>7.2f}% | "
              f"{vm['f1']*100:>7.2f}% | {vm['precision']*100:>6.2f}% | "
              f"{vm['recall']*100:>6.2f}% | {time.time()-t0:>5.1f}s")

        if vm["iou"] > best_val_iou:
            best_val_iou = vm["iou"]
            torch.save({
                "arch": student_arch,
                "in_channels": student_in_channels,
                "channel_names": None,
                "loss": args.loss,
                "norm": args.norm,
                "scale": args.scale,
                "epoch": epoch,
                "teacher": args.teacher,
                "model_state_dict": model.state_dict(),
                "val_iou": vm["iou"],
                "val_f1": vm["f1"],
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
            }, os.path.join(args.output_dir, "best_model.pth"))

    # --- Step 5: one test evaluation of the best-val student ---------------
    print("-" * len(header))
    print(f"Self-training complete in {(time.time()-t_start)/60:.1f} min. "
          f"Best val IoU {best_val_iou*100:.2f}%")
    best = torch.load(os.path.join(args.output_dir, "best_model.pth"),
                      map_location=device, weights_only=False)
    model = get_model(arch=best["arch"],
                      in_channels=best["in_channels"]).to(device)
    model.load_state_dict(best["model_state_dict"])
    _, tm = evaluate(model, test_loader, criterion, device)
    print("\n" + "=" * 70)
    print("Student on REAL test labels (transductive: test images were seen "
          "unlabelled in training):")
    print("=" * 70)
    print(f"  Test IoU       : {tm['iou'] * 100:.2f}%")
    print(f"  Test F1 / Dice : {tm['f1'] * 100:.2f}%")
    print(f"  Test Precision : {tm['precision'] * 100:.2f}%")
    print(f"  Test Recall    : {tm['recall'] * 100:.2f}%")

    with open(os.path.join(args.output_dir, "training_history.json"),
              "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nArtifacts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
