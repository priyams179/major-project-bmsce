"""Training loop for Siamese U-Net landslide model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import LandslidePairs, list_events, split_events
from .model import DiceBCELoss, SiameseUNet, metrics

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def train(
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-4,
    patch: int = 128,
    stride: int = 64,
    base: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    events = list_events()
    train_ev, val_ev = split_events(events, val_frac=0.2, seed=seed)
    print(f"Train events ({len(train_ev)}): {train_ev}")
    print(f"Val events ({len(val_ev)}): {val_ev}")

    train_ds = LandslidePairs(train_ev, patch, stride, augment=True)
    val_ds = LandslidePairs(val_ev, patch, stride, augment=False)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    in_ch = int(train_ds[0]["x"].shape[0])
    model = SiameseUNet(in_ch=in_ch, base=base).to(device)
    print(f"Model input channels: {in_ch}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = DiceBCELoss(pos_weight=5.0)

    best_f1 = 0.0
    best_path = CKPT_DIR / "best.pt"
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_m = 0.0, []
        for batch in train_dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            v = batch["valid"].to(device)
            # Skip batches with no valid pixels or no signal.
            if v.sum() < 100:
                continue
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y, v)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            tr_m.append(metrics(logits.detach(), y, v))
        sched.step()

        model.eval()
        va_loss, va_m = 0.0, []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                v = batch["valid"].to(device)
                logits = model(x)
                va_loss += criterion(logits, y, v).item()
                va_m.append(metrics(logits, y, v))

        def avg(ms, key):
            # Aggregate from summed confusion counts for stable F1.
            tp = sum(m["tp"] for m in ms)
            fp = sum(m["fp"] for m in ms)
            fn = sum(m["fn"] for m in ms)
            tn = sum(m["tn"] for m in ms)
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "acc": acc}

        tr_s = avg(tr_m, "train")
        va_s = avg(va_m, "val")
        n_tr = len(train_ds)
        n_va = len(val_ds)
        dt = time.time() - t0
        print(
            f"Epoch {ep:02d}/{epochs} ({dt:.0f}s) "
            f"loss {tr_loss/max(len(train_dl),1):.4f}/{va_loss/max(len(val_dl),1):.4f} "
            f"| train F1 {tr_s['f1']:.3f} | val F1 {va_s['f1']:.3f} "
            f"P {va_s['precision']:.3f} R {va_s['recall']:.3f} IoU {va_s['iou']:.3f}"
        )
        history.append({"epoch": ep, "train": tr_s, "val": va_s})

        if va_s["f1"] > best_f1:
            best_f1 = va_s["f1"]
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "f1": best_f1,
                "base": base,
                "in_ch": in_ch,
                "train_events": train_ev,
                "val_events": val_ev,
            }, best_path)
            print(f"  -> saved best (F1 {best_f1:.3f})")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val F1: {best_f1:.3f} -> {best_path}")
    return best_path
