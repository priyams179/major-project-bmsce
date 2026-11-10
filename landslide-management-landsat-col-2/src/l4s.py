"""Landslide4Sense benchmark: honest real-ground-truth evaluation.

Train/eval a U-Net on L4S h5 patches (14 bands, 128x128, 10 m). Masks exist
only for TrainData (3799); the official valid/test sets ship no public masks,
so we carve an honest split out of the labelled set ourselves.

Honesty protocol (fixes the earlier leaky 80/20 + val-tuned threshold):
  * THREE-way split 70/15/15 (seed 42): train / val / test.
  * Model selection (best epoch) uses VAL only.
  * The decision threshold is swept on VAL only and then frozen.
  * The headline F1 is reported on the held-out TEST set, which is touched
    exactly once, at the frozen threshold. Nothing is tuned on it.

Note on residual leakage: the public L4S patches carry no geo-coordinates, so
a strict geographic hold-out is impossible from the data alone. The 3-way
split removes model- and threshold-selection leakage (the previously reported
0.754 @ thr 0.65 was the best threshold on the *reported* set); any remaining
patch-adjacency leakage is shared with every published split-based L4S number.

Normalization: per-band mean/std computed on the TRAIN split only (cached to
checkpoints/l4s_norm.json). The previously hard-coded constants did not match
the data on disk (true band means ~0.9-1.7, old constants ~-0.49..0.78), which
left inputs far from zero-centred.

Format: TrainData/img/image_N.h5 -> hf['img'] (H,W,14) float; mask (H,W) {0,1}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import (
    DeepUNet,
    DiceBCELoss,
    FocalTverskyBCELoss,
    SingleUNet,
    metrics,
    tta_logits,
)

L4S_DIR = Path("data/L4S")
CKPT_DIR = Path("checkpoints")
NORM_PATH = CKPT_DIR / "l4s_norm.json"

# Fallback (official baseline) constants, kept only for the legacy/baseline run.
OFFICIAL_MEAN = np.array(
    [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644,
     0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819],
    dtype=np.float32,
)
OFFICIAL_STD = np.array(
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


def split_l4s_3way(root: Path = L4S_DIR, val_frac: float = 0.15,
                   test_frac: float = 0.15, seed: int = 42):
    """Deterministic train/val/test split of the labelled L4S patches."""
    pairs = _list_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No L4S pairs under {root} (TrainData extracted?)")
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n = len(pairs)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = [pairs[i] for i in idx[:n_test]]
    val = [pairs[i] for i in idx[n_test:n_test + n_val]]
    train = [pairs[i] for i in idx[n_test + n_val:]]
    return train, val, test


def compute_norm(pairs: list[tuple[Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over the given (train) patches, NaN-safe."""
    s = np.zeros(14, np.float64)
    ss = np.zeros(14, np.float64)
    npx = 0
    for ip, _ in pairs:
        with h5py.File(ip, "r") as hf:
            a = np.nan_to_num(hf["img"][:].astype(np.float64)).reshape(-1, 14)
        s += a.sum(0)
        ss += (a * a).sum(0)
        npx += a.shape[0]
    mean = s / npx
    std = np.sqrt(np.maximum(ss / npx - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class L4SPatches(Dataset):
    def __init__(self, pairs, mean, std, augment: bool = False):
        self.pairs = pairs
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        with h5py.File(ip, "r") as hf:
            img = np.nan_to_num(hf["img"][:].astype(np.float32))  # [H,W,14]
        with h5py.File(mp, "r") as hf:
            mask = hf["mask"][:].astype(np.float32)               # [H,W]

        x = img.transpose(2, 0, 1)             # [14,H,W]
        x = (x - self.mean) / self.std
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
                x = x * (1.0 + np.random.uniform(-0.1, 0.1))

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


@torch.no_grad()
def _collect_probs(model, dl, device, use_tta: bool):
    """Return flat (probs, gts) over a loader, optionally with TTA."""
    model.eval()
    ps, gs = [], []
    for batch in dl:
        x = batch["x"].to(device)
        logits = tta_logits(model, x) if use_tta else model(x)
        ps.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        gs.append(batch["y"].numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(gs)


def _f1_at(P: np.ndarray, G: np.ndarray, thr: float) -> dict:
    pred = P > thr
    pos = G > 0.5
    tp = float((pred & pos).sum())
    fp = float((pred & ~pos).sum())
    fn = float((~pred & pos).sum())
    tn = float((~pred & ~pos).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {"thr": round(float(thr), 3), "f1": f1, "precision": prec,
            "recall": rec, "iou": iou, "acc": acc}


def train_l4s(
    epochs: int = 90,
    batch_size: int = 16,
    lr: float = 3e-4,
    base: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    improved: bool = True,
    warmup: int = 3,
) -> dict:
    """Train + honestly evaluate the L4S model.

    improved=True  -> DeepUNet, train-derived normalization, Focal-Tversky
                      loss, warmup+cosine LR, TTA at eval (the shipped model).
    improved=False -> legacy SingleUNet + official constants + DiceBCE, no TTA
                      (an honest baseline on the same split, for comparison).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)
    tag = "improved" if improved else "baseline"
    print(f"Device: {device} | mode: {tag}")

    tr_pairs, va_pairs, te_pairs = split_l4s_3way(seed=seed)
    print(f"L4S split (seed {seed}): {len(tr_pairs)} train / "
          f"{len(va_pairs)} val / {len(te_pairs)} test (real human GT)")

    if improved:
        mean, std = compute_norm(tr_pairs)  # train-only stats, no leakage
        with open(NORM_PATH, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist(),
                       "source": "train_split"}, f, indent=2)
        print(f"Normalization: per-band train stats -> {NORM_PATH}")
    else:
        mean, std = OFFICIAL_MEAN, OFFICIAL_STD
        print("Normalization: legacy official constants")

    train_ds = L4SPatches(tr_pairs, mean, std, augment=True)
    val_ds = L4SPatches(va_pairs, mean, std, augment=False)
    test_ds = L4SPatches(te_pairs, mean, std, augment=False)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    if improved:
        model = DeepUNet(in_ch=14, base=base).to(device)
        criterion = FocalTverskyBCELoss(pos_weight=5.0, alpha=0.3, beta=0.7)
    else:
        model = SingleUNet(in_ch=14, base=base).to(device)
        criterion = DiceBCELoss(pos_weight=5.0)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {type(model).__name__} base={base} ({n_par:.1f}M params)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):  # linear warmup then cosine decay
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt = CKPT_DIR / f"best_l4s{'' if improved else '_baseline'}.pt"
    best = {"f1": -1.0}
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

        # Plain (no-TTA) val F1 @ 0.5 for cheap per-epoch model selection.
        model.eval()
        va_m = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                va_m.append(metrics(model(x), y, thr=0.5))
        tr_s, va_s = _agg(tr_m), _agg(va_m)
        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} ({dt:.0f}s) lr {opt.param_groups[0]['lr']:.1e} "
              f"| train F1 {tr_s['f1']:.3f} | val F1 {va_s['f1']:.3f} "
              f"P {va_s['precision']:.3f} R {va_s['recall']:.3f} IoU {va_s['iou']:.3f}")
        history.append({"epoch": ep, "train": tr_s, "val": va_s})

        if va_s["f1"] > best["f1"]:
            best = dict(va_s, epoch=ep)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "f1": va_s["f1"], "base": base, "in_ch": 14,
                        "arch": type(model).__name__, "improved": improved,
                        "mean": mean.tolist(), "std": std.tolist(),
                        "dataset": "landslide4sense"}, ckpt)
            print(f"  -> saved best (val F1 {va_s['f1']:.3f}, epoch {ep})")

    # Reload best-epoch weights for the final honest evaluation.
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    use_tta = improved

    # 1) Freeze the threshold by sweeping VAL only.
    Pv, Gv = _collect_probs(model, val_dl, device, use_tta)
    sweep = [_f1_at(Pv, Gv, t) for t in np.arange(0.20, 0.81, 0.05)]
    best_val = max(sweep, key=lambda s: s["f1"])
    frozen_thr = best_val["thr"]

    # 2) Report on the held-out TEST set at thr 0.5 and at the frozen thr.
    Pt, Gt = _collect_probs(model, test_dl, device, use_tta)
    test_050 = _f1_at(Pt, Gt, 0.5)
    test_frozen = _f1_at(Pt, Gt, frozen_thr)

    print("\n=== VAL threshold sweep (used only to freeze the threshold) ===")
    for s in sweep:
        mark = "  <== frozen" if s is best_val else ""
        print(f"  thr {s['thr']:.2f}: F1 {s['f1']:.3f} "
              f"P {s['precision']:.3f} R {s['recall']:.3f}{mark}")
    print("\n=== HELD-OUT TEST (real human GT, touched once) ===")
    print(f"  thr 0.50        : F1 {test_050['f1']:.4f} "
          f"P {test_050['precision']:.3f} R {test_050['recall']:.3f} "
          f"IoU {test_050['iou']:.3f} acc {test_050['acc']:.4f}")
    print(f"  thr {frozen_thr:.2f} (val) : F1 {test_frozen['f1']:.4f} "
          f"P {test_frozen['precision']:.3f} R {test_frozen['recall']:.3f} "
          f"IoU {test_frozen['iou']:.3f} acc {test_frozen['acc']:.4f}")
    print(f"\nHEADLINE (honest) F1 = {test_frozen['f1']:.4f} "
          f"on {len(te_pairs)} held-out test patches, thr {frozen_thr:.2f} "
          f"(TTA={use_tta})")

    suffix = "" if improved else "_baseline"
    with open(CKPT_DIR / f"history_l4s{suffix}.json", "w") as f:
        json.dump(history, f, indent=2)
    report = {
        "mode": tag, "tta": use_tta, "seed": seed,
        "n_train": len(tr_pairs), "n_val": len(va_pairs), "n_test": len(te_pairs),
        "best_epoch": best["epoch"], "val_best": best_val,
        "frozen_threshold": frozen_thr,
        "test_thr050": test_050, "test_frozen": test_frozen,
        "val_sweep": sweep,
    }
    with open(CKPT_DIR / f"l4s_report{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    # Back-compat artifact names referenced by the report.
    with open(CKPT_DIR / f"l4s_val_metrics{suffix}.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(CKPT_DIR / f"l4s_sweep{suffix}.json", "w") as f:
        json.dump({"val_sweep": sweep, "frozen": best_val,
                   "test": test_frozen}, f, indent=2)
    print(f"Artifacts -> {CKPT_DIR}/l4s_report{suffix}.json (+ sweep/val/history)")
    return report
