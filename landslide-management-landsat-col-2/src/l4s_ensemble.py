"""L4S ensemble evaluation: average the improved DeepUNet and the legacy
SingleUNet (each with its own normalization + TTA) on the same honest 3-way
split. Diverse-model averaging is the cheapest reliable way past the single
U-Net ceiling. Threshold is frozen on VAL, headline F1 reported once on TEST.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .l4s import CKPT_DIR, L4SPatches, _f1_at, split_l4s_3way
from .model import DeepUNet, SingleUNet, tta_logits


def _load(ckpt_name: str, cls, device):
    st = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    m = cls(in_ch=14, base=st.get("base", 64)).to(device).eval()
    m.load_state_dict(st["model"])
    mean = np.array(st["mean"], np.float32)
    std = np.array(st["std"], np.float32)
    return m, mean, std


@torch.no_grad()
def _ensemble_probs(models_norms, pairs, device, batch_size=16):
    """Mean TTA probability over models (each fed its own normalization)."""
    loaders = [
        DataLoader(L4SPatches(pairs, mean, std, augment=False),
                   batch_size, shuffle=False, num_workers=4, pin_memory=True)
        for _, mean, std in models_norms
    ]
    all_p, all_g = [], []
    for batches in zip(*loaders):
        gts = batches[0]["y"].numpy().reshape(-1)
        acc = None
        for (model, _, _), batch in zip(models_norms, batches):
            x = batch["x"].to(device)
            p = torch.sigmoid(tta_logits(model, x)).cpu().numpy().reshape(-1)
            acc = p if acc is None else acc + p
        all_p.append(acc / len(models_norms))
        all_g.append(gts)
    return np.concatenate(all_p), np.concatenate(all_g)


def eval_ensemble(seed: int = 42) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, va_pairs, te_pairs = split_l4s_3way(seed=seed)

    m_imp = _load("best_l4s.pt", DeepUNet, device)
    m_base = _load("best_l4s_baseline.pt", SingleUNet, device)
    models_norms = [m_imp, m_base]
    print(f"Ensemble: DeepUNet + SingleUNet | {len(va_pairs)} val / {len(te_pairs)} test")

    Pv, Gv = _ensemble_probs(models_norms, va_pairs, device)
    sweep = [_f1_at(Pv, Gv, t) for t in np.arange(0.20, 0.81, 0.05)]
    best_val = max(sweep, key=lambda s: s["f1"])
    thr = best_val["thr"]

    Pt, Gt = _ensemble_probs(models_norms, te_pairs, device)
    test_050 = _f1_at(Pt, Gt, 0.5)
    test_frozen = _f1_at(Pt, Gt, thr)

    print("\n=== VAL sweep (freeze threshold) ===")
    for s in sweep:
        mark = "  <== frozen" if s is best_val else ""
        print(f"  thr {s['thr']:.2f}: F1 {s['f1']:.3f} "
              f"P {s['precision']:.3f} R {s['recall']:.3f}{mark}")
    print("\n=== HELD-OUT TEST (ensemble, touched once) ===")
    print(f"  thr 0.50        : F1 {test_050['f1']:.4f} "
          f"P {test_050['precision']:.3f} R {test_050['recall']:.3f} "
          f"IoU {test_050['iou']:.3f} acc {test_050['acc']:.4f}")
    print(f"  thr {thr:.2f} (val) : F1 {test_frozen['f1']:.4f} "
          f"P {test_frozen['precision']:.3f} R {test_frozen['recall']:.3f} "
          f"IoU {test_frozen['iou']:.3f} acc {test_frozen['acc']:.4f}")
    print(f"\nENSEMBLE HEADLINE F1 = {test_frozen['f1']:.4f} @ thr {thr:.2f}")

    report = {"members": ["DeepUNet", "SingleUNet"], "frozen_threshold": thr,
              "val_best": best_val, "test_thr050": test_050,
              "test_frozen": test_frozen, "val_sweep": sweep,
              "n_val": len(va_pairs), "n_test": len(te_pairs)}
    with open(CKPT_DIR / "l4s_ensemble_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Artifact -> {CKPT_DIR}/l4s_ensemble_report.json")
    return report


if __name__ == "__main__":
    eval_ensemble()
