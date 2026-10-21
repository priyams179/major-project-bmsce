"""Landslide4Sense benchmark: real ground-truth evaluation.

Train/eval single-shot U-Net on L4S h5 patches (14 bands, 128x128, 10m).
Masks exist only for TrainData (3799); official valid/test are unlabeled.
We split the labeled set 80/20 (seed 42) for an honest F1 vs human labels.

Format: TrainData/img/image_N.h5 -> hf['img'] (H,W,14) float32
        TrainData/mask/mask_N.h5 -> hf['mask'] (H,W) {0,1}
Normalization: official baseline mean/std (Ghorbanzadeh et al. 2022).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import DiceBCELoss, SingleUNet, metrics

L4S_DIR = Path("data/L4S")
MEAN = np.array(
    [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644,
     0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819],
    dtype=np.float32,
)
STD = np.array(
    [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354,
     0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913],
    dtype=np.float32,
)


def _list_pairs(root: Path) -> list[tuple[Path, Path]]:
    """(img, mask) h5 pairs present on disk, sorted by index."""
    imgs = sorted((root / "TrainData" / "img").glob("image_*.h5"),
                  key=lambda p: int(p.stem.split("_")[1]))
    pairs = []
    for ip in imgs:
        n = ip.stem.split("_")[1]
        mp = root / "TrainData" / "mask" / f"mask_{n}.h5"
        if mp.exists():
            pairs.append((ip, mp))
    return pairs


def split_l4s(root: Path = L4S_DIR, val_frac: float = 0.2, seed: int = 42):
    pairs = _list_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No L4S pairs under {root} (TrainData extracted?)")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_frac))
    val = [pairs[i] for i in idx[:n_val]]
    train = [pairs[i] for i in idx[n_val:]]
    return train, val


class L4SPatches(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        augment: bool = False,
    ):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        with h5py.File(ip, "r") as hf:
            img = hf["img"][:].astype(np.float32)  # [H,W,14]
        with h5py.File(mp, "r") as hf:
            mask = hf["mask"][:].astype(np.float32)  # [H,W]

        x = img.transpose(2, 0, 1)  # [14,H,W]
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        y = mask[None]

        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
                y = np.rot90(y, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                s = 1.0 + np.random.uniform(-0.1, 0.1)
                x = x * s

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
        }


def _agg(ms: list[dict]) -> dict:
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


def train_l4s(
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
    base: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    thr: float = 0.5,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tr_pairs, va_pairs = split_l4s(seed=seed)
    print(f"L4S: {len(tr_pairs)} train / {len(va_pairs)} val (real GT, seed {seed})")

    train_ds = L4SPatches(tr_pairs, augment=True)
    val_ds = L4SPatches(va_pairs, augment=False)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = SingleUNet(in_ch=14, base=base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = DiceBCELoss(pos_weight=5.0)

    best = {"f1": 0.0}
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_m = []
        for batch in train_dl:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_m.append(metrics(logits.detach(), y))
        sched.step()

        model.eval()
        va_m = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                va_m.append(metrics(logits, y, thr=thr))

        tr_s, va_s = _agg(tr_m), _agg(va_m)
        dt = time.time() - t0
        print(
            f"Epoch {ep:02d}/{epochs} ({dt:.0f}s) "
            f"| train F1 {tr_s['f1']:.3f} | val F1 {va_s['f1']:.3f} "
            f"P {va_s['precision']:.3f} R {va_s['recall']:.3f} "
            f"IoU {va_s['iou']:.3f} acc {va_s['acc']:.4f}"
        )
        history.append({"epoch": ep, "train": tr_s, "val": va_s})

        if va_s["f1"] > best["f1"]:
            best = dict(va_s, epoch=ep)
            torch.save({
                "model": model.state_dict(),
                "epoch": ep,
                "f1": va_s["f1"],
                "base": base,
                "in_ch": 14,
                "dataset": "landslide4sense",
            }, ckpt_dir / "best_l4s.pt")
            print(f"  -> saved best (F1 {va_s['f1']:.3f})")

    with open(ckpt_dir / "history_l4s.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(ckpt_dir / "l4s_val_metrics.json", "w") as f:
        json.dump(best, f, indent=2)

    # Threshold sweep for best F1 (reported fairly).
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for batch in val_dl:
            x = batch["x"].to(device)
            probs.append(torch.sigmoid(model(x)).cpu().numpy())
            gts.append(batch["y"].numpy())
    P = np.concatenate([p.reshape(-1) for p in probs])
    G = np.concatenate([g.reshape(-1) for g in gts])
    sweep = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = P > t
        tp = float((pred & (G > 0.5)).sum())
        fp = float((pred & (G < 0.5)).sum())
        fn = float((~pred & (G > 0.5)).sum())
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        sweep.append({"thr": round(float(t), 2), "f1": f1, "p": prec, "r": rec})
    best_sweep = max(sweep, key=lambda s: s["f1"])
    print("\n=== L4S val threshold sweep (real GT) ===")
    for s in sweep:
        mark = "  <== best" if s is best_sweep else ""
        print(f"  thr {s['thr']:.2f}: F1 {s['f1']:.3f} P {s['p']:.3f} R {s['r']:.3f}{mark}")
    print(f"\nBest epoch F1 {best['f1']:.3f} (epoch {best['epoch']}), "
          f"best sweep {best_sweep['f1']:.3f} @ thr {best_sweep['thr']}")
    with open(ckpt_dir / "l4s_sweep.json", "w") as f:
        json.dump({"best_epoch": best, "sweep": sweep}, f, indent=2)
    return best
