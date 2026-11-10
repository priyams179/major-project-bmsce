"""Download imagery for the landslide pipeline from NASA Earthdata.

NASA Earthdata serves **HLS** (Harmonized Landsat-Sentinel, 30 m COG GeoTIFFs),
*not* ESA `.SAFE`. HLS carries the same optical bands (B01-B12) the models use,
so `run_scene.py --hls <folder>` ingests it directly. For a true `.SAFE`
(10 m), download from Copernicus CDSE instead (see --help notes).

Auth: a free Earthdata Login (https://urs.earthdata.nasa.gov). earthaccess reads
~/.netrc, or env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD, or prompts.

Examples
--------
  # HLS Sentinel-2 over Wayanad around the 2024-07-30 event:
  python download_earthdata.py --hls --bbox 76.0 11.3 76.3 11.6 \
      --start 2024-07-25 --end 2024-08-10 --out data/scenes

  # NASADEM elevation tile for the slope/DEM bands:
  python download_earthdata.py --nasadem --bbox 76.0 11.3 76.3 11.6 --out data/dem
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--hls", action="store_true", help="download HLSS30 (Sentinel-2 HLS)")
    what.add_argument("--hls-landsat", action="store_true", help="download HLSL30 (Landsat HLS)")
    what.add_argument("--nasadem", action="store_true", help="download NASADEM elevation")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LONMIN", "LATMIN", "LONMAX", "LATMAX"))
    ap.add_argument("--start", help="YYYY-MM-DD (HLS only)")
    ap.add_argument("--end", help="YYYY-MM-DD (HLS only)")
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--max", type=int, default=3, help="max granules to download")
    args = ap.parse_args()

    import earthaccess
    earthaccess.login()  # ~/.netrc, env vars, or interactive

    if args.nasadem:
        short_name, kw = "NASADEM_HGT", {}
    elif args.hls_landsat:
        short_name, kw = "HLSL30", {"temporal": (args.start, args.end)}
    else:
        short_name, kw = "HLSS30", {"temporal": (args.start, args.end)}

    print(f"Searching {short_name} over {args.bbox} ...")
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=tuple(args.bbox),
        count=args.max,
        **{k: v for k, v in kw.items() if v and all(v)},
    )
    if not results:
        print("No granules found. Widen the bbox/date range or check the product.")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(results)} granule(s) -> {out} ...")
    files = earthaccess.download(results, str(out))
    print(f"Done: {len(files)} files in {out}")
    if not args.nasadem:
        print("Next: group the per-band GeoTIFFs into one granule folder, then run\n"
              f"  python run_scene.py --hls {out}/<granule> --out results/scene")


if __name__ == "__main__":
    main()
