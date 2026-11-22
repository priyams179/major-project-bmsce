"""Data preparation: download pre/post Sentinel-2 composites for each event.

Writes ``data/landsentry/<event>/{pre,post}.tif`` -- int16 GeoTIFFs with bands
B2, B3, B4, B8 (TOA reflectance x 1e4) + n_obs (clear observations), all on a
shared UTM 43N 10 m grid.

Usage:  python -m landsentry prepare [--event kodagu|wayanad|all]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rasterio

from . import config as C
from . import gee


def write_tif(path, arr, transform, crs, nodata=None, descriptions=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                count=arr.shape[0], dtype=arr.dtype, crs=crs, transform=transform,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr)
        if descriptions:
            for i, s in enumerate(descriptions, 1):
                d.set_band_description(i, s)


def download_event(name: str) -> None:
    ev = C.EVENTS[name]
    out = C.event_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    gee.init()
    import ee
    region = ee.Geometry.Rectangle(list(ev.bbox))
    bands = list(C.BANDS) + ["n_obs"]
    meta = {"event": name, "event_date": ev.date, "bbox": ev.bbox,
            "collection": C.S2_COLLECTION, "cloud_prob_max": C.CLOUD_PROB_MAX,
            "bands": bands, "epsg": C.UTM_EPSG}
    for tag, (d0, d1) in (("pre", ev.pre), ("post", ev.post)):
        t0 = time.time()
        n_img = (ee.ImageCollection(C.S2_COLLECTION).filterBounds(region)
                 .filterDate(d0, d1)
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.SCENE_CLOUD_MAX))
                 .size().getInfo())
        print(f"[{name}] {tag} {d0}..{d1}: {n_img} S2 scenes -> composite", flush=True)
        arr, tfm, crs = gee.download(
            gee.s2_composite(region, d0, d1), bands, ev.bbox, epsg=C.UTM_EPSG,
            progress=lambda i, n: print(f"    tile {i}/{n}", flush=True) if i % 4 == 0 or i == n else None)
        write_tif(out / f"{tag}.tif", arr, tfm, crs, descriptions=bands)
        valid = (arr[:4] != 0).all(0)
        meta[tag] = {"window": [d0, d1], "n_scenes": n_img,
                     "shape": list(arr.shape[1:]),
                     "valid_frac": float(valid.mean()),
                     "median_n_obs": float(np.median(arr[4][valid])) if valid.any() else 0}
        print(f"[{name}] {tag}: {arr.shape[1]}x{arr.shape[2]} px, valid "
              f"{valid.mean()*100:.1f}%, median clear obs {meta[tag]['median_n_obs']:.0f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    (out / "imagery.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="all", choices=["all", *C.EVENTS])
    a = ap.parse_args(argv)
    for name in (C.EVENTS if a.event == "all" else [a.event]):
        download_event(name)


if __name__ == "__main__":
    main()
