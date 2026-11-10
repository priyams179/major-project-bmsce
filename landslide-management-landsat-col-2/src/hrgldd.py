"""HR-GLDD segmentation: real-mask landslide segmentation on a dataset that
includes the Indian Western Ghats (Kodagu, Karnataka, 2018).

Dataset: Meena et al. 2023, ESSD (HR-GLDD), Zenodo 10.5281/zenodo.7189381,
CC BY 4.0. PlanetScope 3 m, 4 bands (R,G,B,NIR), 128x128 patches, with an
author-provided train/val/test split (6 .npy files in data/HR_GLDD/).

Why this dataset: it is the only publicly downloadable landslide-segmentation
set with *real digitized pixel masks* over an Indian site. High-resolution
(3 m) real masks give much sharper, higher-F1 segmentation than the 10-30 m
pseudo-labelled Landsat events.

Protocol (same honesty rules as src/l4s.py):
  * Author-provided train/val/test split (no leakage introduced by us).
  * Per-band normalization from TRAIN only.
  * Best epoch + decision threshold frozen on VAL; reported ONCE on TEST.
  * D4 test-time augmentation at eval.

Note: the public .npy split is globally mixed (Kodagu tiles are not separately
labelled in the release), so the held-out metric is a global HR-GLDD figure on
a set that includes the Indian Western Ghats, not a Kodagu-only score.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import DeepUNet, FocalTverskyBCELoss, metrics, tta_logits

HR_DIR = Path("data/HR_GLDD")
CKPT_DIR = Path("checkpoints")


def _load_xy(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, Y) for a split as X:[N,C,H,W] float32, Y:[N,1,H,W] {0,1}."""
    X = np.load(HR_DIR / f"{split}X.npy")
    Y = np.load(HR_DIR / f"{split}Y.npy")
    # Normalize channel order to [N,C,H,W]. Channels = the small axis (3/4/5).
    if X.ndim == 4 and X.shape[-1] in (3, 4, 5, 6) and X.shape[1] not in (3, 4, 5, 6):
        X = np.transpose(X, (0, 3, 1, 2))
    # Masks -> [N,1,H,W]
    if Y.ndim == 3:
        Y = Y[:, None]
    elif Y.ndim == 4 and Y.shape[-1] == 1:
        Y = np.transpose(Y, (0, 3, 1, 2))
    elif Y.ndim == 4 and Y.shape[1] != 1:
        Y = Y[:, :1]
    Y = (Y > 0.5).astype(np.float32)
    return np.ascontiguousarray(X.astype(np.float32)), np.ascontiguousarray(Y)


def _add_ndvi(X: np.ndarray) -> np.ndarray:
    """Append NDVI = (NIR-R)/(NIR+R) as an extra channel (bands R,G,B,NIR)."""
    r, nir = X[:, 0:1], X[:, 3:4]
    ndvi = (nir - r) / (nir + r + 1e-6)
    return np.concatenate([X, ndvi.astype(np.float32)], axis=1)


class HRGLDDSet(Dataset):
    def __init__(self, X, Y, mean, std, augment=False):
        self.X, self.Y = X, Y
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        y = self.Y[i]
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
                x = x * (1.0 + np.random.uniform(-0.1, 0.1))
        return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                "y": torch.from_numpy(np.ascontiguousarray(y))}


def _agg(ms):
    tp = sum(m["tp"] for m in ms); fp = sum(m["fp"] for m in ms)
    fn = sum(m["fn"] for m in ms); tn = sum(m["tn"] for m in ms)
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return {"precision": prec, "recall": rec, "f1": f1,
            "iou": tp / (tp + fp + fn + 1e-6),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-6)}


def _f1_at(P, G, thr):
    pred = P > thr; pos = G > 0.5
    tp = float((pred & pos).sum()); fp = float((pred & ~pos).sum())
    fn = float((~pred & pos).sum()); tn = float((~pred & ~pos).sum())
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    return {"thr": round(float(thr), 3), "f1": f1, "precision": prec,
            "recall": rec, "iou": tp / (tp + fp + fn + 1e-6),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-6)}


@torch.no_grad()
def _collect(model, dl, device, tta=True):
    model.eval(); ps, gs = [], []
    for b in dl:
        x = b["x"].to(device)
        logit = tta_logits(model, x) if tta else model(x)
        ps.append(torch.sigmoid(logit).cpu().numpy().reshape(-1))
        gs.append(b["y"].numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(gs)


def train_hrgldd(epochs=80, batch_size=16, lr=3e-4, base=64, num_workers=4,
                 seed=42, warmup=3, use_ndvi=True) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {device}")

    trX, trY = _load_xy("train")
    vaX, vaY = _load_xy("val")
    teX, teY = _load_xy("test")
    if use_ndvi:
        trX, vaX, teX = _add_ndvi(trX), _add_ndvi(vaX), _add_ndvi(teX)
    in_ch = trX.shape[1]
    print(f"HR-GLDD: train {trX.shape} / val {vaX.shape} / test {teX.shape} "
          f"| in_ch={in_ch} (ndvi={use_ndvi})")
    print(f"pos pixel rate: train {trY.mean():.4f} / val {vaY.mean():.4f} "
          f"/ test {teY.mean():.4f}")

    mean = trX.reshape(trX.shape[0], in_ch, -1).mean((0, 2)).astype(np.float32)
    std = (trX.reshape(trX.shape[0], in_ch, -1).std((0, 2)) + 1e-6).astype(np.float32)
    with open(CKPT_DIR / "hrgldd_norm.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(),
                   "in_ch": int(in_ch), "use_ndvi": use_ndvi}, f, indent=2)

    tr_dl = DataLoader(HRGLDDSet(trX, trY, mean, std, augment=True), batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True,
                       drop_last=True)
    va_dl = DataLoader(HRGLDDSet(vaX, vaY, mean, std), batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    te_dl = DataLoader(HRGLDDSet(teX, teY, mean, std), batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    model = DeepUNet(in_ch=in_ch, base=base).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: DeepUNet in_ch={in_ch} base={base} ({n_par:.1f}M)")
    pos_w = max(1.0, float((trY == 0).sum() / max((trY == 1).sum(), 1)) ** 0.5)
    criterion = FocalTverskyBCELoss(pos_weight=pos_w, alpha=0.3, beta=0.7)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(epochs - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt = CKPT_DIR / "best_hrgldd.pt"
    best = {"f1": -1.0}; history = []

    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tr_m = []
        for b in tr_dl:
            x, y = b["x"].to(device), b["y"].to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval(); va_m = []
        with torch.no_grad():
            for b in va_dl:
                x, y = b["x"].to(device), b["y"].to(device)
                va_m.append(metrics(model(x), y, thr=0.5))
        va = _agg(va_m); dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} ({dt:.0f}s) lr {opt.param_groups[0]['lr']:.1e} "
              f"| val F1 {va['f1']:.4f} P {va['precision']:.3f} R {va['recall']:.3f} "
              f"IoU {va['iou']:.3f}")
        history.append({"epoch": ep, "val": va})
        if va["f1"] > best["f1"]:
            best = dict(va, epoch=ep)
            torch.save({"model": model.state_dict(), "epoch": ep, "base": base,
                        "in_ch": in_ch, "mean": mean.tolist(), "std": std.tolist(),
                        "use_ndvi": use_ndvi, "dataset": "hrgldd"}, ckpt)
            print(f"  -> saved best (val F1 {va['f1']:.4f}, epoch {ep})")

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    Pv, Gv = _collect(model, va_dl, device, tta=True)
    sweep = [_f1_at(Pv, Gv, t) for t in np.arange(0.20, 0.81, 0.05)]
    best_val = max(sweep, key=lambda s: s["f1"]); thr = best_val["thr"]
    Pt, Gt = _collect(model, te_dl, device, tta=True)
    test_050 = _f1_at(Pt, Gt, 0.5); test_frozen = _f1_at(Pt, Gt, thr)

    print("\n=== HELD-OUT TEST (HR-GLDD real masks, incl. Indian Western Ghats) ===")
    print(f"  thr 0.50        : F1 {test_050['f1']:.4f} P {test_050['precision']:.3f} "
          f"R {test_050['recall']:.3f} IoU {test_050['iou']:.3f} acc {test_050['acc']:.4f}")
    print(f"  thr {thr:.2f} (val) : F1 {test_frozen['f1']:.4f} P {test_frozen['precision']:.3f} "
          f"R {test_frozen['recall']:.3f} IoU {test_frozen['iou']:.3f} acc {test_frozen['acc']:.4f}")
    print(f"\nHEADLINE HR-GLDD segmentation F1 = {test_frozen['f1']:.4f} "
          f"IoU {test_frozen['iou']:.4f} @ thr {thr:.2f}")

    report = {"dataset": "HR-GLDD", "in_ch": int(in_ch), "use_ndvi": use_ndvi,
              "best_epoch": best["epoch"], "frozen_threshold": thr,
              "val_best": best_val, "test_thr050": test_050,
              "test_frozen": test_frozen, "val_sweep": sweep,
              "n_test_patches": int(len(teX))}
    with open(CKPT_DIR / "hrgldd_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(CKPT_DIR / "history_hrgldd.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Artifacts -> {CKPT_DIR}/hrgldd_report.json")
    return report


if __name__ == "__main__":
    train_hrgldd()
