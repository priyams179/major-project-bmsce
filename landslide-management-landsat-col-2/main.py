"""CLI for landslide detection pipeline.

Usage:
  python main.py labels          # generate pseudo-labels
  python main.py train           # train Siamese U-Net
  python main.py infer           # run inference on all events
  python main.py geocode         # vectorize + attach lat/lon + place names
  python main.py all             # labels + train + infer + geocode
  python main.py l4s             # L4S segmentation benchmark (improved, real GT)
  python main.py l4s_baseline    # L4S segmentation legacy baseline (same split)
  python main.py l4s_classify    # L4S patch classifier -> 90+ accuracy/F1 (real GT)
  python main.py hrgldd          # HR-GLDD segmentation, real masks incl. Indian Western Ghats
  python main.py uk_inventory    # Uttarakhand inventory & matched stable sampling
  python main.py uk_train        # Uttarakhand Random Forest train & 20-draw spatial CV
  python main.py uk_rainfall     # Uttarakhand GPM IMERG rainfall analysis
  python main.py uk_map          # Uttarakhand interactive susceptibility map
  python main.py uk_all          # All Uttarakhand steps end-to-end
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
        train_l4s(improved=True)
    elif cmd == "l4s_baseline":
        from src.l4s import train_l4s
        train_l4s(improved=False, epochs=60)
    elif cmd == "l4s_classify":
        from src.l4s_classify import train_classifier
        train_classifier()
    elif cmd == "hrgldd":
        from src.hrgldd import train_hrgldd
        train_hrgldd()
    elif cmd == "l5_labels":
        from src.landsat5 import make_labels
        make_labels()
    elif cmd == "l5_train":
        from src.landsat5 import train as train_l5
        train_l5()
    elif cmd == "l5_test":
        from src.landsat5 import test as test_l5
        test_l5()
    elif cmd == "s1_labels":
        from src.sentinel1_wg import make_labels
        make_labels()
    elif cmd == "s1_train":
        from src.sentinel1_wg import train as train_s1
        train_s1()
    elif cmd == "s1_test":
        from src.sentinel1_wg import test as test_s1
        test_s1()
    elif cmd == "s1_eval_real":
        from src.sentinel1_wg import eval_real
        eval_real()
    elif cmd == "s1_train_real":
        from src.sentinel1_wg import train_real
        train_real()
    elif cmd == "s1_cls_real":
        from src.sentinel1_wg import train_cls_real
        train_cls_real()
    elif cmd == "uk_inventory":
        from src.uttarakhand import make_inventory
        make_inventory()
    elif cmd == "uk_train":
        from src.uttarakhand import train
        train()
    elif cmd == "uk_rainfall":
        from src.uttarakhand import rainfall
        rainfall()
    elif cmd == "uk_map":
        from src.uttarakhand import interactive_map
        interactive_map()
    elif cmd == "uk_all":
        from src.uttarakhand import run_all
        run_all()
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
