#!/usr/bin/env python3
"""
Cross-architecture ensemble evaluation with scale promotion and TTA.

Averages probability maps of several diverse checkpoints (different
architectures / training recipes). The decision threshold is chosen on
val only; the selected ensemble is then evaluated and reported on test.

Usage:
    python ensemble_eval.py --members models/m5_selftrain_r2/best_model.pth \
        models/m4_dlv3p_s256/best_model.pth models/m4_resnet_s256/best_model.pth
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from models import get_model
from train_segmentation import LandslideDataset, compute_or_load_split_stats


def load_member_probs(ckpt_path: str, data_dir: str, split: str,
                      target_scale: int = 2, tta: bool = True):
    """Rebuild a checkpoint's model with its own preprocessing and predict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_model(
        arch=ck["arch"],
        in_channels=ck["in_channels"],
        base_features=ck.get("base_features", 32),
        decoder_dropout=ck.get("decoder_dropout", 0.0),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    norm = ck.get("norm", "none")
    scale = ck.get("scale", 1)
    stats = compute_or_load_split_stats(data_dir) if norm != "none" else None
    ds = LandslideDataset(data_dir, split=split, augment=False,
                          norm=norm, scale=scale, stats=stats)
    probs = []
    target_hw = 128 * target_scale
    with torch.no_grad():
        for i in range(len(ds)):
            img_t, _, _ = ds[i]
            x = img_t.unsqueeze(0).to(device)
            if tta:
                x_h = torch.flip(x, dims=[-1])
                x_v = torch.flip(x, dims=[-2])
                x_hv = torch.flip(x, dims=[-2, -1])
                p0 = torch.sigmoid(model(x))
                ph = torch.flip(torch.sigmoid(model(x_h)), dims=[-1])
                pv = torch.flip(torch.sigmoid(model(x_v)), dims=[-2])
                phv = torch.flip(torch.sigmoid(model(x_hv)), dims=[-2, -1])
                p = ((p0 + ph + pv + phv) / 4.0)[0, 0]
            else:
                p = torch.sigmoid(model(x))[0, 0]

            if p.shape[-1] != target_hw:
                p = F.interpolate(p.unsqueeze(0).unsqueeze(0),
                                  size=(target_hw, target_hw),
                                  mode="bilinear", align_corners=False)[0, 0]
            probs.append(p.cpu().numpy().astype(np.float32))

    del model
    torch.cuda.empty_cache()
    return np.stack(probs)


def metrics(P: np.ndarray, Y: np.ndarray, V: np.ndarray, t: float):
    b = (P >= t) & (V > 0.5)
    g = (Y > 0.5) & (V > 0.5)
    tp = (b & g).sum()
    fp = (b & ~g).sum()
    fn = (~b & g).sum()
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return iou, f1, prec, rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--members", nargs="+", required=True,
                   help="Checkpoint paths to ensemble")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Optional relative weights for each member")
    p.add_argument("--data-dir", default="data/processed/l4s")
    p.add_argument("--scale", type=int, default=2,
                   help="Target spatial scale factor (default: 2 -> 256x256)")
    p.add_argument("--no-tta", action="store_true",
                   help="Disable test-time augmentation")
    args = p.parse_args()

    use_tta = not args.no_tta
    weights = args.weights
    if weights is not None:
        assert len(weights) == len(args.members), "Number of weights must match members"
        norm_w = [w / sum(weights) for w in weights]
    else:
        norm_w = [1.0 / len(args.members)] * len(args.members)

    print(f"Loading ground truth masks (scale x{args.scale})...")
    val_ds = LandslideDataset(args.data_dir, split="val", augment=False,
                              norm="none", scale=args.scale)
    test_ds = LandslideDataset(args.data_dir, split="test", augment=False,
                               norm="none", scale=args.scale)

    Yv = np.stack([val_ds[i][1][0].numpy() > 0.5 for i in range(len(val_ds))])
    Vv = np.stack([val_ds[i][2][0].numpy() > 0.5 for i in range(len(val_ds))])
    Yt = np.stack([test_ds[i][1][0].numpy() > 0.5 for i in range(len(test_ds))])
    Vt = np.stack([test_ds[i][2][0].numpy() > 0.5 for i in range(len(test_ds))])

    print(f"\nEnsembling {len(args.members)} members (TTA={use_tta}, target scale x{args.scale}):")
    Pv_sum = np.zeros_like(Yv, dtype=np.float32)
    Pt_sum = np.zeros_like(Yt, dtype=np.float32)

    for idx, path in enumerate(args.members):
        name = path.replace("/best_model.pth", "").replace("models/", "")
        w = norm_w[idx]
        print(f"[{idx+1}/{len(args.members)}] Evaluating {name} (weight {w:.2f})...")
        Pv = load_member_probs(path, args.data_dir, "val", target_scale=args.scale, tta=use_tta)
        Pt = load_member_probs(path, args.data_dir, "test", target_scale=args.scale, tta=use_tta)

        # Single member score on val and test at default 0.5
        v_iou, v_f1, _, _ = metrics(Pv, Yv, Vv, 0.5)
        t_iou, t_f1, _, _ = metrics(Pt, Yt, Vt, 0.5)
        print(f"  {name:25s} | val IoU@0.5={v_iou*100:.2f}% (F1 {v_f1*100:.2f}%) | test IoU@0.5={t_iou*100:.2f}% (F1 {t_f1*100:.2f}%)")

        Pv_sum += Pv * w
        Pt_sum += Pt * w
        del Pv, Pt

    # Threshold chosen on val only, then applied once to test.
    grid = np.arange(0.20, 0.81, 0.02)
    best_t = max(grid, key=lambda t: metrics(Pv_sum, Yv, Vv, t)[0])
    vIoU, vF1, vP, vR = metrics(Pv_sum, Yv, Vv, best_t)
    tIoU, tF1, tP, tR = metrics(Pt_sum, Yt, Vt, best_t)

    print("\n" + "=" * 70)
    print(f"FINAL ENSEMBLE ({len(args.members)} models) BENCHMARK REPORT:")
    print("=" * 70)
    print(f"  Optimal Val Threshold : {best_t:.2f}")
    print(f"  Validation IoU        : {vIoU*100:.2f}%")
    print(f"  Validation F1         : {vF1*100:.2f}%  (Prec: {vP*100:.1f}%, Rec: {vR*100:.1f}%)")
    print(f"  Test IoU              : {tIoU*100:.2f}%")
    print(f"  Test F1 / Dice        : {tF1*100:.2f}%  (Prec: {tP*100:.1f}%, Rec: {tR*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
