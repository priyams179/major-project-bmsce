"""Ensemble inference: model probability AND multi-index consensus.

High-precision mask = model_prob > 0.5  ∧  multi_index_mask == 1
(or soft blend). Also computes val F1 of each component for QC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from .dataset import (
    DATA_DIR,
    NODATA,
    extract_landsat_bands,
    list_events,
    load_pair,
    normalize_stack,
    split_events,
    valid_mask,
)
from .indices import multi_index_mask
from .model import SiameseUNet

OUT_DIR = Path("results/masks_ens")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def f1_from_masks(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    p = pred & valid
    g = gt.astype(bool) & valid
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "acc": acc,
            "tp": tp, "fp": fp, "fn": fn}


@torch.no_grad()
def ensemble_all(
    checkpoint: Path = Path("checkpoints/best.pt"),
    thr: float = 0.30,
    tile: int = 128,
    overlap: int = 32,
    require_both: bool = True,
    soft_thr: float = 0.15,
    mode: str = "strict",
) -> None:
    """mode='strict':   ens = (prob > thr) & index            (P-first)
    mode='relaxed': ens = (prob > thr) & (index | prob > 0.55)  (recall)"""
    if mode == "relaxed":
        out_dir = Path("results/masks_ens_relaxed")
        thr_model = 0.25        # lower gate: keep more candidates
        thr_conf = 0.55         # confident model pixels pass without index
    else:
        out_dir = Path("results/masks_ens")
        thr_model = thr
        thr_conf = 1.1         # unreachable -> strict behaviour
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SiameseUNet(
        in_ch=ckpt.get("in_ch", 9), base=ckpt.get("base", 64)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    events = list_events()
    _, val_ev = split_events(events, val_frac=0.2, seed=42)
    val_set = set(val_ev)

    val_stats = {"model": [], "index": [], "ens": []}
    mask_dir = Path("data/pseudo_masks_v2")

    for eid in events:
        pre, post, prof = load_pair(eid)
        h, w = pre.shape[1:]

        pre_n = normalize_stack(pre)
        post_n = normalize_stack(post)
        diff = pre_n - post_n
        full = np.concatenate([pre_n, post_n, diff], 0)
        valid = valid_mask(pre, post, NODATA)

        acc = np.zeros((h, w), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        step = tile - overlap
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        for y in ys:
            for x in xs:
                patch = full[:, y : y + tile, x : x + tile]
                vp = valid[y : y + tile, x : x + tile]
                ph, pw = patch.shape[1:]
                if ph < tile or pw < tile:
                    patch = np.pad(patch, ((0, 0), (0, tile - ph), (0, tile - pw)))
                    vp = np.pad(vp, ((0, tile - vp.shape[0]), (0, tile - vp.shape[1])))
                t = torch.from_numpy(patch[None].astype(np.float32))
                t = t * torch.from_numpy(vp[None, None].astype(np.float32))
                logits = model(t.to(device))
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                acc[y : y + ph, x : x + pw] += prob[:ph, :pw]
                cnt[y : y + ph, x : x + pw] += 1.0
        cnt[cnt == 0] = 1
        prob_map = acc / cnt

        model_m = (prob_map > thr_model) & valid
        idx_m = multi_index_mask(
            extract_landsat_bands(pre), extract_landsat_bands(post), NODATA
        ).astype(bool)
        # Soft blend: boost prob where index agrees.
        blended = prob_map * np.where(idx_m, 1.5, 1.0)
        soft_m = (blended > soft_thr) & valid
        if mode == "relaxed":
            ens_m = model_m & (idx_m | (prob_map > thr_conf))
        elif require_both:
            ens_m = model_m & idx_m
        else:
            ens_m = soft_m | (model_m & idx_m)

        # Save ensemble mask.
        out = ens_m.astype(np.uint8)
        profile = prof.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
        with rasterio.open(out_dir / f"{eid}_mask.tif", "w", **profile) as dst:
            dst.write(out, 1)

        # Val metrics vs multi-index pseudo-labels.
        if eid in val_set:
            gt = idx_m
            val_stats["model"].append(f1_from_masks(model_m, gt, valid))
            val_stats["index"].append(f1_from_masks(idx_m, gt, valid))
            val_stats["ens"].append(f1_from_masks(ens_m, gt, valid))

    def agg(ms):
        tp = sum(m["tp"] for m in ms)
        fp = sum(m["fp"] for m in ms)
        fn = sum(m["fn"] for m in ms)
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        acc = (tp + tp) / (tp + tp + fp + fn + 1e-6)  # placeholder, recompute
        return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}

    print("\n=== Val metrics vs multi-index pseudo-labels ===")
    for name in ("model", "index", "ens"):
        s = agg(val_stats[name])
        print(
            f"  {name:6s}: F1 {s['f1']:.3f}  P {s['precision']:.3f}  "
            f"R {s['recall']:.3f}  IoU {s['iou']:.3f}"
        )
    print(f"\nEnsemble masks ({mode}) -> {out_dir}")

    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump({k: agg(v) for k, v in val_stats.items()}, f, indent=2)


if __name__ == "__main__":
    import sys
    ensemble_all(mode=sys.argv[1] if len(sys.argv) > 1 else "strict")
