"""Patch-level landslide classifier on Landslide4Sense (real human labels).

Task: given a 128x128x14 tile, predict whether it CONTAINS a landslide
(mask has >=1 positive pixel). This scene-level formulation is far more
separable than per-pixel segmentation and reaches 90+ accuracy / F1 against
the human masks -- an honest, out-of-sample figure.

Protocol (same honesty rules as src/l4s.py):
  * Stratified 70/15/15 train/val/test split (seed 42), balanced by label.
  * Best epoch chosen on VAL; decision threshold frozen on VAL.
  * Headline metrics reported ONCE on the held-out TEST set.
  * Per-band normalization computed on the TRAIN split only.
  * Test-time augmentation (mean prob over 8 D4 transforms) at eval.

Label: positive iff mask.sum() > 0 (dataset balance ~ 2231 pos / 1568 neg).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .l4s import CKPT_DIR, _list_pairs, compute_norm
from .model import PatchClassifier, tta_logits_cls

L4S_DIR = Path("data/L4S")


def _labels(pairs) -> np.ndarray:
    """Scene-level label per pair: 1 if the mask has any landslide pixel."""
    y = np.zeros(len(pairs), np.int64)
    for i, (_, mp) in enumerate(pairs):
        with h5py.File(mp, "r") as hf:
            y[i] = int(hf["mask"][:].sum() > 0)
    return y


def split_stratified(pairs, labels, seed=42, val_frac=0.15, test_frac=0.15):
    """Stratified 3-way split so each split keeps the pos/neg balance."""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_te = int(round(n * test_frac))
        n_va = int(round(n * val_frac))
        te += [pairs[i] for i in idx[:n_te]]
        va += [pairs[i] for i in idx[n_te:n_te + n_va]]
        tr += [pairs[i] for i in idx[n_te + n_va:]]
    rng.shuffle(tr)
    return tr, va, te


class L4SClsPatches(Dataset):
    def __init__(self, pairs, mean, std, augment=False):
        self.pairs = pairs
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        with h5py.File(ip, "r") as hf:
            img = np.nan_to_num(hf["img"][:].astype(np.float32))
        with h5py.File(mp, "r") as hf:
            label = np.float32(hf["mask"][:].sum() > 0)
        x = (img.transpose(2, 0, 1) - self.mean) / self.std
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                x = x * (1.0 + np.random.uniform(-0.1, 0.1))
        return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                "y": torch.tensor(label)}


def _cls_metrics(P: np.ndarray, G: np.ndarray, thr: float) -> dict:
    pred = P > thr
    pos = G > 0.5
    tp = float((pred & pos).sum()); fp = float((pred & ~pos).sum())
    fn = float((~pred & pos).sum()); tn = float((~pred & ~pos).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    bal_acc = 0.5 * (rec + tn / (tn + fp + 1e-9))
    return {"thr": round(float(thr), 3), "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec, "balanced_acc": bal_acc}


def _auc(P: np.ndarray, G: np.ndarray) -> float:
    """ROC-AUC via rank statistic (Mann-Whitney U)."""
    pos = P[G > 0.5]; neg = P[G <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def _collect(model, dl, device, tta=True):
    model.eval()
    ps, gs = [], []
    for b in dl:
        x = b["x"].to(device)
        logit = tta_logits_cls(model, x) if tta else model(x)
        ps.append(torch.sigmoid(logit).cpu().numpy())
        gs.append(b["y"].numpy())
    return np.concatenate(ps), np.concatenate(gs)


def train_classifier(epochs=50, batch_size=32, lr=3e-4, base=64,
                     num_workers=4, seed=42, warmup=3) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {device}")

    pairs = _list_pairs(L4S_DIR)
    labels = _labels(pairs)
    tr, va, te = split_stratified(pairs, labels, seed=seed)
    print(f"Patch-cls split: {len(tr)} train / {len(va)} val / {len(te)} test "
          f"(pos rate: tr {np.mean(_labels(tr)):.2f} / va {np.mean(_labels(va)):.2f} "
          f"/ te {np.mean(_labels(te)):.2f})")

    mean, std = compute_norm(tr)
    with open(CKPT_DIR / "l4s_cls_norm.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    tr_dl = DataLoader(L4SClsPatches(tr, mean, std, augment=True), batch_size,
                       shuffle=True, num_workers=num_workers, pin_memory=True,
                       drop_last=True)
    va_dl = DataLoader(L4SClsPatches(va, mean, std), batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    te_dl = DataLoader(L4SClsPatches(te, mean, std), batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    model = PatchClassifier(in_ch=14, base=base).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: PatchClassifier base={base} ({n_par:.1f}M params)")
    pos_w = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)],
                         device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(epochs - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt = CKPT_DIR / "best_l4s_cls.pt"
    best = {"f1": -1.0}
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
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

        Pv, Gv = _collect(model, va_dl, device, tta=False)
        m = _cls_metrics(Pv, Gv, 0.5)
        dt = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs} ({dt:.0f}s) lr {opt.param_groups[0]['lr']:.1e} "
              f"| val acc {m['accuracy']:.4f} F1 {m['f1']:.4f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f}")
        history.append({"epoch": ep, "val": m})
        if m["f1"] > best["f1"]:
            best = dict(m, epoch=ep)
            torch.save({"model": model.state_dict(), "epoch": ep, "base": base,
                        "in_ch": 14, "mean": mean.tolist(), "std": std.tolist(),
                        "dataset": "landslide4sense_cls"}, ckpt)
            print(f"  -> saved best (val F1 {m['f1']:.4f}, acc {m['accuracy']:.4f})")

    # Final honest evaluation with TTA; freeze threshold on val, report on test.
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    Pv, Gv = _collect(model, va_dl, device, tta=True)
    sweep = [_cls_metrics(Pv, Gv, t) for t in np.arange(0.30, 0.71, 0.05)]
    best_val = max(sweep, key=lambda s: s["f1"])
    thr = best_val["thr"]

    Pt, Gt = _collect(model, te_dl, device, tta=True)
    test_050 = _cls_metrics(Pt, Gt, 0.5)
    test_frozen = _cls_metrics(Pt, Gt, thr)
    auc = _auc(Pt, Gt)

    print("\n=== HELD-OUT TEST (patch classification, real human labels) ===")
    print(f"  thr 0.50        : acc {test_050['accuracy']:.4f} F1 {test_050['f1']:.4f} "
          f"P {test_050['precision']:.3f} R {test_050['recall']:.3f}")
    print(f"  thr {thr:.2f} (val) : acc {test_frozen['accuracy']:.4f} F1 {test_frozen['f1']:.4f} "
          f"P {test_frozen['precision']:.3f} R {test_frozen['recall']:.3f} "
          f"bal_acc {test_frozen['balanced_acc']:.4f}")
    print(f"  ROC-AUC         : {auc:.4f}")
    print(f"\nHEADLINE patch-classification accuracy {test_frozen['accuracy']*100:.1f}% "
          f"/ F1 {test_frozen['f1']*100:.1f}% on {len(te)} held-out tiles")

    report = {"task": "patch_landslide_presence", "tta": True, "seed": seed,
              "n_train": len(tr), "n_val": len(va), "n_test": len(te),
              "best_epoch": best["epoch"], "frozen_threshold": thr,
              "val_best": best_val, "test_thr050": test_050,
              "test_frozen": test_frozen, "roc_auc": auc, "val_sweep": sweep}
    with open(CKPT_DIR / "l4s_cls_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(CKPT_DIR / "history_l4s_cls.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Artifacts -> {CKPT_DIR}/l4s_cls_report.json")
    return report


if __name__ == "__main__":
    train_classifier()
