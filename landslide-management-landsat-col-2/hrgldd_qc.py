"""Qualitative HR-GLDD segmentation panels: RGB | ground truth | prediction.

Usage: python hrgldd_qc.py [n_examples]
Loads checkpoints/best_hrgldd.pt, runs it on held-out TEST patches (with TTA),
and writes qc/hrgldd_<i>.png for the patches with the clearest landslides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.hrgldd import _add_ndvi, _load_xy
from src.model import DeepUNet, tta_logits

OUT = Path("qc")
OUT.mkdir(exist_ok=True)


def main(n: int = 6) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load("checkpoints/best_hrgldd.pt", map_location=device,
                    weights_only=False)
    teX, teY = _load_xy("test")
    rgb = teX[:, :3].transpose(0, 2, 3, 1).copy()  # R,G,B for display
    X = _add_ndvi(teX) if st.get("use_ndvi") else teX
    mean = np.array(st["mean"], np.float32)[:, None, None]
    std = np.array(st["std"], np.float32)[:, None, None]
    thr = 0.5

    model = DeepUNet(in_ch=st["in_ch"], base=st["base"]).to(device).eval()
    model.load_state_dict(st["model"])

    # Pick the test patches with the most landslide pixels (clear examples).
    order = np.argsort(-teY.reshape(len(teY), -1).sum(1))
    picks = order[:n]
    for rank, i in enumerate(picks):
        x = torch.from_numpy(((X[i] - mean) / std)[None]).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(tta_logits(model, x))[0, 0].cpu().numpy()
        pred = prob > thr
        gt = teY[i, 0] > 0.5
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)

        disp = np.clip(rgb[i] / (np.percentile(rgb[i], 99) + 1e-6), 0, 1)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
        ax[0].imshow(disp); ax[0].set_title(f"HR-GLDD test #{i} — RGB")
        ax[1].imshow(disp); ax[1].imshow(gt, cmap="Reds", alpha=0.45)
        ax[1].set_title("ground truth (human mask)")
        ax[2].imshow(disp); ax[2].imshow(pred, cmap="Reds", alpha=0.45)
        ax[2].set_title(f"prediction — patch F1 {f1:.2f}")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        out = OUT / f"hrgldd_{rank:02d}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {out} (patch #{i}, F1 {f1:.2f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
