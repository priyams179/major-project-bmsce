"""CLI for landslide detection pipeline.

Usage:
  python main.py labels          # generate pseudo-labels
  python main.py train           # train Siamese U-Net
  python main.py infer           # run inference on all events
  python main.py geocode         # vectorize + attach lat/lon + place names
  python main.py all             # labels + train + infer + geocode
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.dataset import NODATA, extract_landsat_bands, list_events, load_pair
from src.indices import multi_index_mask


def make_labels() -> None:
    out = Path("data/pseudo_masks_v2")
    out.mkdir(parents=True, exist_ok=True)
    events = list_events()
    print(f"Generating multi-index pseudo-labels for {len(events)} events...")
    for eid in events:
        pre, post, _ = load_pair(eid)
        m = multi_index_mask(
            extract_landsat_bands(pre), extract_landsat_bands(post), NODATA
        )
        np_path = out / f"{eid}.npy"
        np.save(np_path, m)
        frac = m.mean()
        print(f"  {eid}: {frac*100:.2f}% landslide pixels")
    print(f"Saved to {out}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd == "labels":
        make_labels()
    elif cmd == "train":
        from src.train import train
        train()
    elif cmd == "infer":
        from src.infer import infer_all
        infer_all()
    elif cmd == "geocode":
        from src.geocode import geocode_results
        geocode_results()
    elif cmd == "l4s":
        from src.l4s import train_l4s
        train_l4s()
    elif cmd == "all":
        make_labels()
        from src.train import train
        train()
        from src.infer import infer_all
        infer_all()
        from src.geocode import geocode_results
        geocode_results()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
