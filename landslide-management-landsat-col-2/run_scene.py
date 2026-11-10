"""End-to-end landslide inference on a satellite scene.

Takes a Sentinel-2 .SAFE (or HLS folder, or pre-stacked GeoTIFF), runs the
trained classifier + segmentation models, and writes georeferenced outputs.

Examples
--------
  # Sentinel-2 .SAFE (uploaded from Copernicus CDSE), with a DEM, AOI crop:
  python run_scene.py --safe data/scenes/S2B_..._.SAFE \
      --dem data/dem/kerala_dem.tif \
      --bbox 76.0 11.3 76.3 11.6 --out results/scene_wayanad

  # NASA Earthdata HLS granule folder:
  python run_scene.py --hls data/scenes/HLS.S30.T43PFS.2024213 --out results/hls

  # Pre-stacked 14-band GeoTIFF (bands already in L4S order):
  python run_scene.py --geotiff my_stack.tif --out results/custom

Flags: --gate (keep segmentation only inside classifier-positive tiles),
       --norm {scene,l4s}, --stride N, --seg-thr F, --cls-thr F, --no-tta.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--safe", help="Sentinel-2 .SAFE directory")
    src.add_argument("--hls", help="NASA Earthdata HLS granule folder")
    src.add_argument("--geotiff", help="pre-stacked 14-band GeoTIFF (L4S order)")
    ap.add_argument("--dem", help="DEM GeoTIFF for slope/elevation bands")
    ap.add_argument("--bbox", nargs=4, type=float,
                    metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"),
                    help="AOI crop in WGS84")
    ap.add_argument("--out", default="results/scene", help="output directory")
    ap.add_argument("--norm", choices=["scene", "l4s"], default="scene")
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seg-thr", type=float, default=0.65)
    ap.add_argument("--cls-thr", type=float, default=0.35)
    ap.add_argument("--gate", action="store_true",
                    help="mask segmentation to classifier-positive tiles only")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    from src.scene import read_geotiff, read_hls, read_safe
    from src.predict_scene import load_models, predict_scene, save_outputs

    print("Reading scene...")
    if args.safe:
        stack, profile, valid = read_safe(args.safe, args.dem, tuple(args.bbox) if args.bbox else None)
    elif args.hls:
        stack, profile, valid = read_hls(args.hls, args.dem, tuple(args.bbox) if args.bbox else None)
    else:
        stack, profile, valid = read_geotiff(args.geotiff)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | norm={args.norm} | gate={args.gate} | tta={not args.no_tta}")

    # Stats for --norm l4s come from the segmentation checkpoint.
    seg_stats = None
    if args.norm == "l4s":
        import torch as _t
        s = _t.load(Path("checkpoints/best_l4s.pt"), map_location="cpu", weights_only=False)
        seg_stats = {"mean": s["mean"], "std": s["std"]}

    print("Running classification + segmentation...")
    res = predict_scene(stack, profile, valid, norm=args.norm, stride=args.stride,
                        gate=args.gate, tta=not args.no_tta, seg_thr=args.seg_thr,
                        cls_thr=args.cls_thr, device=device, seg_ckpt_stats=seg_stats)
    summary = save_outputs(res, args.out, stack=stack)
    print(f"\nDone. {summary['n_polygons']} landslide polygons, "
          f"{summary['mapped_area_km2']} km² -> {args.out}/")


if __name__ == "__main__":
    main()
