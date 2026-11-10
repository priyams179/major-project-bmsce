"""Visualise the Western Ghats SAR **tile-level** classifier (92.9% acc) on the
real Kodagu scene: SAR background with every tile the classifier flags as
landslide shaded with diagonal stripes, plus the real inventory polygons so you
can see predicted-tile vs truth.

Output: qc/s1_kodagu_tiles.png   Usage: python s1_tile_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle

from src.model import PatchClassifier, tta_logits_cls
from src.sentinel1_wg import PATCH, _cls_stack, _load_gt, _norm, _valid

OUT = Path("qc/s1_kodagu_tiles.png")
THR = 0.45          # frozen operating threshold from training
POS_PX = 20         # a tile is "truth-positive" if it holds >=20 inventory px


def _fc(stack):
    """SAR false-colour: R=VV, G=VH, B=VV-VH (dB -> 0..1)."""
    vv = np.clip((stack[0] + 25) / 35, 0, 1)
    vh = np.clip((stack[1] + 30) / 30, 0, 1)
    return np.clip(np.stack([vv, vh, np.clip(vv - vh + 0.3, 0, 1)], -1), 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load("checkpoints/best_s1_wg_cls.pt", map_location=device,
                    weights_only=False)
    model = PatchClassifier(in_ch=st["in_ch"], base=st["base"]).to(device).eval()
    model.load_state_dict(st["model"])
    pre, post, mask = _load_gt()
    H, W = pre.shape[1:]

    # Classify every non-overlapping tile; collect flagged tiles + whether the
    # tile truly contains inventory landslide (for the legend tally).
    flagged, tp, fp = [], 0, 0
    with torch.no_grad():
        for y in range(0, H - PATCH + 1, PATCH):
            for x in range(0, W - PATCH + 1, PATCH):
                pp, qp = pre[:, y:y+PATCH, x:x+PATCH], post[:, y:y+PATCH, x:x+PATCH]
                if _valid(pp, qp).mean() < 0.5:
                    continue
                xin = torch.from_numpy(_cls_stack(pp, qp)[None].astype(np.float32)).to(device)
                prob = float(torch.sigmoid(tta_logits_cls(model, xin))[0].cpu())
                if prob >= THR:
                    truth = mask[y:y+PATCH, x:x+PATCH].sum() >= POS_PX
                    flagged.append((y, x, truth))
                    tp += int(truth); fp += int(not truth)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(_fc(post))
    # real inventory polygons (blue) underneath
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="winter", alpha=0.55, zorder=2)
    # flagged tiles: yellow stripes = correct (overlaps truth), magenta = false alarm
    for y, x, truth in flagged:
        col = (1, 1, 0, 0.9) if truth else (1, 0, 1, 0.9)
        ax.add_patch(Rectangle((x, y), PATCH, PATCH, facecolor=(*col[:3], 0.08),
                               edgecolor=col, hatch="////", linewidth=1.0, zorder=3))
    ax.set_title(f"Kodagu 2018 — SAR tile classifier (92.9% acc): "
                 f"{tp} correct / {fp} false-alarm flagged tiles")
    ax.legend(handles=[
        Patch(facecolor="blue", alpha=0.55, label="real inventory landslide (truth)"),
        Patch(facecolor=(1, 1, 0, 0.2), edgecolor="y", hatch="////",
              label=f"tile flagged landslide — correct ({tp})"),
        Patch(facecolor=(1, 0, 1, 0.2), edgecolor="m", hatch="////",
              label=f"tile flagged — false alarm ({fp})")],
        loc="lower right", fontsize=8, framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT} ({len(flagged)} flagged tiles: {tp} correct, {fp} false-alarm)")


if __name__ == "__main__":
    main()
