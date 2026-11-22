"""Evaluate one checkpoint on one event.

Usage:
  python -m landsentry evaluate --checkpoint checkpoints/landsentry/siamese_kodagu_s0.pt \
      --event wayanad [--split test|val|train|all] [--threshold 0.5] [--tta]

Without ``--threshold`` the val-tuned threshold stored in the checkpoint is used.
"""

from __future__ import annotations

import argparse
import json

from . import config as C
from .data import load_scene
from .infer import evaluate, load_checkpoint, predict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--event", required=True, choices=list(C.EVENTS))
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args(argv)
    net, meta = load_checkpoint(a.checkpoint)
    thr = a.threshold if a.threshold is not None else meta["threshold"]
    s = load_scene(a.event)
    prob = predict(net, s.x, tta=a.tta)
    m = evaluate(prob, s.mask, s.region(a.split), thr=thr)
    print(f"{meta['model']} (trained on {meta['train_on']}, epoch {meta['epoch']}) "
          f"-> {a.event}/{a.split}")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                     indent=2))


if __name__ == "__main__":
    main()
