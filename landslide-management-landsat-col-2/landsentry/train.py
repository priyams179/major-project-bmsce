"""Train a bitemporal change detector on one event (or both).

Usage:
  python -m landsentry train --model siamese --train-on kodagu --seed 0
  python -m landsentry train --model transformer --train-on wayanad

Model selection uses the *source* event's validation region only: the
checkpoint with the best val F1 (at its val-tuned threshold) is kept, and that
threshold is stored in the checkpoint and reused unchanged on every test set,
including the other event -- the target event is never used for tuning.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import models
from .data import CropSampler, load_scene
from .infer import evaluate, predict


class BCEDiceLoss(nn.Module):
    """0.5 * BCE(pos_weight) + 0.5 * soft Dice, both restricted to valid px."""

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pw = pos_weight

    def forward(self, logits, target, valid):
        pw = torch.tensor(self.pw, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
        bce = (bce * valid).sum() / valid.sum().clamp_min(1)
        p = torch.sigmoid(logits) * valid
        t = target * valid
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return 0.5 * bce + 0.5 * dice


def _bbox(region, pad=32):
    ys, xs = np.nonzero(region)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, region.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, region.shape[1])
    return slice(y0, y1), slice(x0, x1)


def validate(net, scenes, split="val", tile=256):
    probs, masks = [], []
    for s in scenes:
        reg = s.region(split)
        if not reg.any():
            continue
        sy, sx = _bbox(reg)
        p = predict(net, s.x[:, sy, sx], tile=tile)
        r = reg[sy, sx]
        probs.append(p[r]); masks.append(s.mask[sy, sx][r])
    prob, mask = np.concatenate(probs), np.concatenate(masks)
    return evaluate(prob, mask, np.ones_like(mask, bool))


def train(model="siamese", train_on="kodagu", epochs=40, steps=200, batch=16, crop=128,
          lr=3e-4, wd=1e-4, pos_weight=5.0, patience=10, seed=0, run_name=None,
          scenes_cache=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(C.EVENTS) if train_on == "both" else [train_on]
    cache = scenes_cache if scenes_cache is not None else {}
    scenes = [cache.setdefault(n, load_scene(n)) for n in names]
    ds = CropSampler(scenes, crop=crop, length=steps * batch, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=True)

    net = models.build(model).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps,
                                                pct_start=0.1)
    loss_fn = BCEDiceLoss(pos_weight)
    run = run_name or f"{model}_{train_on}_s{seed}"
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best, best_ep, hist = -1.0, -1, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        tl = 0.0
        for x, y, v in dl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = net(x)
            loss = loss_fn(logits.float(), y, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item()
        val = validate(net, scenes)
        hist.append({"epoch": ep, "loss": tl / steps, **{k: val[k] for k in
                     ("ap", "best_f1", "best_f1_thr", "f1", "iou")}})
        improved = val["best_f1"] > best
        if improved:
            best, best_ep = val["best_f1"], ep
            torch.save({"model": model, "in_bands": len(C.BANDS), "state": net.state_dict(),
                        "threshold": val["best_f1_thr"], "val": val, "epoch": ep,
                        "train_on": train_on, "seed": seed, "crop": crop,
                        "bands": list(C.BANDS), "norm": "scene_p2_p98_joint"},
                       C.CKPT_DIR / f"{run}.pt")
        if verbose:
            print(f"[{run}] ep {ep:02d} loss {tl/steps:.4f} | val AP {val['ap']:.3f} "
                  f"bestF1 {val['best_f1']:.3f}@{val['best_f1_thr']:.2f} "
                  f"F1@.5 {val['f1']:.3f}{' *' if improved else ''}", flush=True)
        if ep - best_ep >= patience:
            if verbose:
                print(f"[{run}] early stop (no val improvement for {patience} epochs)")
            break
    (C.CKPT_DIR / f"{run}_history.json").write_text(json.dumps(hist, indent=1))
    if verbose:
        print(f"[{run}] best val F1 {best:.3f} @ epoch {best_ep} ({time.time()-t0:.0f}s) "
              f"-> {C.CKPT_DIR / (run + '.pt')}")
    return C.CKPT_DIR / f"{run}.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="siamese", choices=list(models.MODELS))
    ap.add_argument("--train-on", default="kodagu", choices=[*C.EVENTS, "both"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name")
    a = ap.parse_args(argv)
    train(a.model, a.train_on, a.epochs, a.steps, a.batch, a.crop, a.lr,
          pos_weight=a.pos_weight, patience=a.patience, seed=a.seed, run_name=a.run_name)


if __name__ == "__main__":
    main()
