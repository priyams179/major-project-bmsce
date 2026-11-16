"""
Uttarakhand Landslide Risk Mapping — CLI pipeline.

Usage:
    python pipeline.py [--step STEP] [--gee-project PROJECT_ID] [--data-dir DIR] [--out-dir DIR]

Steps:
    1  Build feature raster via GEE   (requires GEE auth + --gee-project)
    2  Build inventory & sample stable points
    3  Run Random Forest model (20-draw spatial CV)
    4  Rainfall analysis
    5  Build interactive map
    all  Run steps 2–5 (default; step 1 must be done separately via GEE)

If the feature raster (data/rudraprayag_features.tif) is absent, steps 2 and 3
will still run using the pre-built training_points_matched_wgs84.geojson.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def _require(step_name: str, exc: Exception) -> None:
    print(f"\n[ERROR] Step '{step_name}' failed: {exc}\n")
    raise SystemExit(1)


def step1_features(args: argparse.Namespace) -> None:
    if not args.gee_project:
        print(
            "Step 1 (features) requires a GEE project ID.\n"
            "Pass --gee-project <your_ee_project_id> and re-run.\n"
            "If you already have rudraprayag_features.tif, copy it to data/ and skip step 1."
        )
        sys.exit(1)
    from src.features import build_features
    build_features(project_id=args.gee_project, drive_folder="landslide_project")
    print("\n>>> Step 1 submitted. Wait for the GEE export task to complete,\n"
          "    then download rudraprayag_features.tif into data/.")


def step2_inventory(args: argparse.Namespace) -> None:
    import pandas as pd  # noqa: F401
    from src.inventory import build_inventory
    try:
        build_inventory(data_dir=args.data_dir, out_dir=args.out_dir)
    except Exception as e:
        _require("inventory", e)


def step3_model(args: argparse.Namespace) -> None:
    from src.model import run_model
    try:
        run_model(data_dir=args.data_dir, out_dir=args.out_dir)
    except Exception as e:
        _require("model", e)


def step4_rainfall(args: argparse.Namespace) -> None:
    from src.rainfall import run_rainfall
    try:
        run_rainfall(data_dir=args.data_dir, out_dir=args.out_dir,
                     gee_project=args.gee_project)
    except Exception as e:
        _require("rainfall", e)


def step5_map(args: argparse.Namespace) -> None:
    from src.interactive_map import build_map
    try:
        build_map(data_dir=args.data_dir, out_dir=args.out_dir)
    except Exception as e:
        _require("interactive_map", e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uttarakhand Landslide Risk Mapping pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which step to run (default: all = steps 2-5)",
    )
    parser.add_argument("--gee-project", default=None,
                        help="Google Earth Engine project ID (needed for steps 1 & 4-download)")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data",
                        help=f"Data directory (default: {ROOT / 'data'})")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs",
                        help=f"Output directory (default: {ROOT / 'outputs'})")
    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Uttarakhand Landslide Risk Mapping")
    print("  Rudraprayag district — Random Forest pipeline")
    print("=" * 60)
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output dir: {args.out_dir}")
    print(f"  Step     : {args.step}")
    print("=" * 60 + "\n")

    t0 = time.time()

    if args.step == "1":
        step1_features(args)
    elif args.step == "2":
        step2_inventory(args)
    elif args.step == "3":
        step3_model(args)
    elif args.step == "4":
        step4_rainfall(args)
    elif args.step == "5":
        step5_map(args)
    else:  # "all"
        print("Running steps 2 → 5 (skipping step 1 — GEE export)")
        print("-" * 60)

        print("\n[Step 2] Building landslide inventory …")
        step2_inventory(args)

        print("\n[Step 3] Running Random Forest model (20-draw) …")
        step3_model(args)

        print("\n[Step 4] Rainfall analysis …")
        step4_rainfall(args)

        print("\n[Step 5] Building interactive map …")
        step5_map(args)

    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f} s")
    print(f"  Figures → {args.out_dir}")


if __name__ == "__main__":
    main()

