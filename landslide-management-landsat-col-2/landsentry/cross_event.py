"""Cross-event generalisation experiment (the LandSentry 2x2 matrix, + "both").

For every model in {siamese, transformer}, training source in
{kodagu, wayanad, both} and seed, train once (selection on the source's val
region only) and test on the held-out *test region* of each event, using the
threshold fixed on the source's val split. Off-diagonal cells (train on one
event, test on the other) are true zero-shot transfers. For those, the
model is additionally scored on the target's *whole scene* (it never saw any
of it).

Outputs (results/landsentry/):
  runs.csv               one row per (model, source, seed, target, region)
  cross_event.json       mean / std per cell
  cross_event.md         paper-style tables
  fig_cross_event.png    2x2 F1 / AP heatmaps per model
  fig_qualitative.png    pre / post / reference / both models on each test set
  fig_training.png       validation curves
  (full-scene probability maps go to data/landsentry/preds/, git-ignored)

Usage:  python -m landsentry cross-event [--seeds 3] [--epochs 40] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product

import numpy as np
import torch

from . import config as C
from .data import load_scene
from .infer import evaluate, load_checkpoint, predict
from .train import train

MODELS = ("siamese", "transformer")
BASELINE = "ndvi_diff"
ALL_METHODS = (BASELINE, *MODELS)
SOURCES = ("kodagu", "wayanad", "both")
TARGETS = ("kodagu", "wayanad")
METRICS = ("f1", "iou", "precision", "recall", "ap", "best_f1")


def baseline_score(scene):
    """Non-learned index baseline: score = clip(NDVI_pre - NDVI_post, 0, 1)."""
    return np.clip(scene.dndvi, 0, 1)


def baseline_rows(scenes, sources):
    """NDVI-differencing baseline under the same protocol (threshold tuned on
    the source event's val region, then frozen)."""
    rows = []
    for src in sources:
        names = list(TARGETS) if src == "both" else [src]
        sp = np.concatenate([baseline_score(scenes[n])[scenes[n].region("val")] for n in names])
        sm = np.concatenate([scenes[n].mask[scenes[n].region("val")] for n in names])
        thr = evaluate(sp, sm, np.ones_like(sm, bool))["best_f1_thr"]
        for tgt in TARGETS:
            s = scenes[tgt]
            regions = [("test", s.region("test"))]
            if src != "both" and src != tgt:
                regions.append(("scene", s.region(None)))
            for rname, reg in regions:
                m = evaluate(baseline_score(s), s.mask, reg, thr=thr)
                rows.append({"model": BASELINE, "source": src, "seed": 0, "target": tgt,
                             "region": rname, "epoch": 0,
                             **{k: m[k] for k in (*METRICS, "threshold", "pos_px", "n_px")}})
            r = rows[-len(regions)]
            print(f"  {BASELINE}_{src} -> {tgt:8s} test: F1 {r['f1']:.3f} AP {r['ap']:.3f} "
                  f"(thr {thr:.2f})", flush=True)
    return rows


def run(seeds=3, epochs=40, resume=False, models=MODELS, sources=SOURCES, tta=False):
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    scenes = {n: load_scene(n) for n in TARGETS}
    rows = baseline_rows(scenes, sources)
    for model, src, seed in product(models, sources, range(seeds)):
        name = f"{model}_{src}_s{seed}"
        ck = C.CKPT_DIR / f"{name}.pt"
        if not (resume and ck.exists()):
            train(model, src, epochs=epochs, seed=seed, run_name=name, scenes_cache=scenes)
        net, meta = load_checkpoint(ck)
        thr = meta["threshold"]
        for tgt in TARGETS:
            s = scenes[tgt]
            prob = predict(net, s.x, tta=tta)
            regions = [("test", s.region("test"))]
            if src != "both" and src != tgt:
                regions.append(("scene", s.region(None)))
            for rname, reg in regions:
                m = evaluate(prob, s.mask, reg, thr=thr)
                rows.append({"model": model, "source": src, "seed": seed, "target": tgt,
                             "region": rname, "epoch": meta["epoch"],
                             **{k: m[k] for k in (*METRICS, "threshold", "pos_px", "n_px")}})
            if seed == 0:
                np.save(C.PRED_DIR / f"prob_{model}_{src}_on_{tgt}.npy",
                        prob.astype(np.float16))
            r = rows[-len(regions)]
            print(f"  {name} -> {tgt:8s} test: F1 {r['f1']:.3f} IoU {r['iou']:.3f} "
                  f"P {r['precision']:.3f} R {r['recall']:.3f} AP {r['ap']:.3f} "
                  f"(thr {thr:.2f})", flush=True)
        del net
        torch.cuda.empty_cache()
    with open(C.RESULTS_DIR / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = summarise(rows)
    (C.RESULTS_DIR / "cross_event.json").write_text(json.dumps(summary, indent=2))
    (C.RESULTS_DIR / "cross_event.md").write_text(markdown(summary, seeds))
    print(markdown(summary, seeds))
    figures(scenes, summary)
    return summary


def summarise(rows):
    out = {}
    keys = sorted({(r["model"], r["source"], r["target"], r["region"]) for r in rows})
    for k in keys:
        sel = [r for r in rows if (r["model"], r["source"], r["target"], r["region"]) == k]
        cell = {"n_seeds": len(sel), "threshold_mean": float(np.mean([r["threshold"] for r in sel]))}
        for m in METRICS:
            v = np.array([r[m] for r in sel], float)
            cell[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0}
        out["|".join(k)] = cell
    return out


def _fmt(c, m):
    return f"{c[m]['mean']:.3f} ± {c[m]['std']:.3f}"


def markdown(summary, seeds):
    L = [f"### Cross-event test-region results (mean ± std over {seeds} seeds; "
         "threshold fixed on the source event's val split; `ndvi_diff` is deterministic)\n",
         "| Model | Train on | Test on | F1 | IoU | Precision | Recall | AP | Oracle F1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for model, src, tgt in product(ALL_METHODS, SOURCES, TARGETS):
        c = summary.get(f"{model}|{src}|{tgt}|test")
        if not c:
            continue
        tag = "" if src in ("both", tgt) else " *(zero-shot)*"
        L.append(f"| {model} | {src} | {tgt}{tag} | " +
                 " | ".join(_fmt(c, m) for m in ("f1", "iou", "precision", "recall", "ap", "best_f1"))
                 + " |")
    L += ["", "### Zero-shot transfer scored on the target's whole scene\n",
          "| Model | Train on | Test on | F1 | IoU | Precision | Recall | AP |",
          "|---|---|---|---|---|---|---|---|"]
    for model, src, tgt in product(ALL_METHODS, ("kodagu", "wayanad"), TARGETS):
        c = summary.get(f"{model}|{src}|{tgt}|scene")
        if c:
            L.append(f"| {model} | {src} | {tgt} | " +
                     " | ".join(_fmt(c, m) for m in ("f1", "iou", "precision", "recall", "ap")) + " |")
    L.append("\n*Oracle F1* = best F1 over all thresholds on the test region itself "
             "(an upper bound that is **not** a valid deployment number; it isolates "
             "ranking quality from threshold miscalibration).")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def _crop_box(mask, region, size=400):
    ys, xs = np.nonzero(mask.astype(bool) & region)
    if ys.size == 0:
        ys, xs = np.nonzero(region)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    y0 = int(np.clip(cy - size // 2, 0, max(mask.shape[0] - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(mask.shape[1] - size, 0)))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def figures(scenes, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) 2x2 heatmaps
    fig, axes = plt.subplots(3, 2, figsize=(10, 12.4))
    for i, model in enumerate(ALL_METHODS):
        for j, met in enumerate(("f1", "ap")):
            ax = axes[i, j]
            M = np.full((2, 2), np.nan)
            S = np.full((2, 2), np.nan)
            for a, src in enumerate(TARGETS):
                for b, tgt in enumerate(TARGETS):
                    c = summary.get(f"{model}|{src}|{tgt}|test")
                    if c:
                        M[a, b], S[a, b] = c[met]["mean"], c[met]["std"]
            ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            for a in range(2):
                for b in range(2):
                    if not np.isnan(M[a, b]):
                        ax.text(b, a, f"{M[a,b]:.3f}\n±{S[a,b]:.3f}", ha="center", va="center",
                                color="white" if M[a, b] < 0.6 else "black", fontsize=12)
            ax.set_xticks([0, 1], [f"test: {t}" for t in TARGETS])
            ax.set_yticks([0, 1], [f"{'tune' if model == BASELINE else 'train'}: {t}"
                                   for t in TARGETS])
            ax.set_title(f"{model} — {'F1 @ val thr' if met == 'f1' else 'Average precision'}")
    fig.suptitle("LandSentry cross-event generalisation (test regions)")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_cross_event.png", dpi=130); plt.close(fig)

    # 2) qualitative
    fig, axes = plt.subplots(2, 7, figsize=(28, 9.6))
    for r, tgt in enumerate(TARGETS):
        s = scenes[tgt]
        sy, sx = _crop_box(s.mask, s.region("test"))
        panels = [("pre", s.rgb_pre[sy, sx]), ("post", s.rgb_post[sy, sx]),
                  ("reference", s.mask[sy, sx])]
        for model in MODELS:
            for src in ("kodagu", "wayanad"):
                p = C.PRED_DIR / f"prob_{model}_{src}_on_{tgt}.npy"
                if p.exists():
                    panels.append((f"{model}\ntrained on {src}",
                                   np.load(p)[sy, sx].astype(np.float32)))
        for k, ax in enumerate(axes[r]):
            ax.axis("off")
            if k >= len(panels):
                continue
            t, im = panels[k]
            if im.ndim == 3:
                ax.imshow(im)
            else:
                ax.imshow(s.rgb_post[sy, sx]); ax.imshow(np.ma.masked_less(im, 0.15), cmap="inferno",
                                                         vmin=0, vmax=1, alpha=0.8)
                ax.contour(s.mask[sy, sx], [0.5], colors="cyan", linewidths=0.6)
            ax.set_title(f"{tgt} test — {t}", fontsize=10)
    fig.suptitle("Change probability (inferno) vs reference outline (cyan); 4 km crops of the test regions")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_qualitative.png", dpi=110); plt.close(fig)

    # 3) training curves (seed 0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, src in zip(axes, SOURCES):
        for model in MODELS:
            h = C.CKPT_DIR / f"{model}_{src}_s0_history.json"
            if h.exists():
                H = json.loads(h.read_text())
                ax.plot([e["epoch"] for e in H], [e["best_f1"] for e in H], label=f"{model} val F1")
                ax.plot([e["epoch"] for e in H], [e["ap"] for e in H], "--", label=f"{model} val AP")
        ax.set_title(f"train on {src} (seed 0)"); ax.set_xlabel("epoch"); ax.set_ylim(0, 1)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_training.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--resume", action="store_true", help="reuse existing checkpoints")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args(argv)
    run(a.seeds, a.epochs, a.resume, tuple(a.models), tuple(a.sources), a.tta)


if __name__ == "__main__":
    main()
