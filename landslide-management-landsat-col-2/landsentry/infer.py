"""Sliding-window full-scene inference and pixel metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import models


def _window(t: int) -> np.ndarray:
    """2-D blending weights that taper to the tile border (avoids seams)."""
    r = np.hanning(t + 2)[1:-1].astype(np.float32)
    w = np.outer(r, r)
    return np.maximum(w, 1e-3)


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray, tile: int = 256, overlap: int = 64,
            batch: int = 16, device: str | None = None, tta: bool = False) -> np.ndarray:
    """Per-pixel change probability for a normalised ``[2C, H, W]`` scene."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    C_, H, W = x.shape
    ph, pw = max(0, tile - H), max(0, tile - W)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    Hp, Wp = x.shape[1:]
    step = tile - overlap
    ys = list(range(0, max(Hp - tile, 0) + 1, step))
    xs = list(range(0, max(Wp - tile, 0) + 1, step))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)
    coords = [(y, x_) for y in ys for x_ in xs]
    acc = np.zeros((Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    win = _window(tile)
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        t = torch.from_numpy(np.stack([x[:, y:y + tile, x_:x_ + tile] for y, x_ in cb])).to(device)
        if tta:
            logit = 0
            for k in range(4):
                for flip in (False, True):
                    a = torch.rot90(t, k, (-2, -1))
                    a = a.flip(-1) if flip else a
                    o = model(a)
                    o = o.flip(-1) if flip else o
                    logit = logit + torch.rot90(o, -k, (-2, -1))
            logit = logit / 8
        else:
            logit = model(t)
        p = torch.sigmoid(logit)[:, 0].float().cpu().numpy()
        for (y, x_), pi in zip(cb, p):
            acc[y:y + tile, x_:x_ + tile] += pi * win
            wsum[y:y + tile, x_:x_ + tile] += win
    return (acc / np.maximum(wsum, 1e-6))[:H, :W]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def counts(prob, mask, region, thr):
    p = (prob >= thr) & region
    g = mask.astype(bool) & region
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    return tp, fp, fn


def scores(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    iou = tp / (tp + fp + fn) if tp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


def evaluate(prob, mask, region, thr=0.5, bins=1000):
    """Metrics at ``thr`` plus threshold-free AP and the best-F1 threshold.

    Uses histograms of the positive / negative probabilities, so the full
    threshold sweep costs one pass over the pixels.
    """
    g = mask.astype(bool)[region]
    pr = prob[region]
    edges = np.linspace(0, 1, bins + 1)
    hp, _ = np.histogram(pr[g], edges)
    hn, _ = np.histogram(pr[~g], edges)
    tp = np.cumsum(hp[::-1])[::-1].astype(np.float64)   # predictions >= edge
    fp = np.cumsum(hn[::-1])[::-1].astype(np.float64)
    P = float(g.sum())
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    rec = tp / max(P, 1)
    f1 = np.where(tp > 0, 2 * tp / (tp + fp + P), 0.0)
    # AP = sum over thresholds of precision * delta-recall (step interpolation)
    r_ext = np.append(rec, 0.0)
    ap = float(np.sum((r_ext[:-1] - r_ext[1:]) * prec))
    b = int(np.argmax(f1))
    out = scores(*counts(prob, mask, region, thr))
    out.update({"threshold": thr, "ap": ap, "best_f1": float(f1[b]),
                "best_f1_thr": float(edges[b]), "pos_px": int(P),
                "pos_frac": P / max(g.size, 1), "n_px": int(g.size)})
    return out


# --------------------------------------------------------------------------- #
def load_checkpoint(path: str | Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    net = models.build(ck["model"], ck.get("in_bands", 4))
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck
