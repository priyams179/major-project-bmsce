"""Build a REAL-ground-truth Sentinel-1 SAR scene for Kodagu 2018:
download S1 VV/VH/slope pre/post over the published landslide inventory extent,
mosaic it, and rasterise the 217 inventory polygons onto the SAR grid.

Inventory: Achaiah et al. (2024), 2018 Kodagu storm, 217 landslide polygons,
CC BY 4.0 (github.com/Arpithaachaiah6/Landslide-inventory-...-Kodagu). This is
*real* ground truth, so src.sentinel1_wg.eval_real can score the SAR model with
a true pixel F1/IoU (not the change-detection pseudo-labels).

Output -> data/Kodagu_SAR_GT/{PRE,POST,mask}.tif  (10 m, UTM; mask 0/1).
Usage:  python build_kodagu_gt.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge

INV = Path("data/kodagu_inventory/Landslide_polygons.shp")
OUT = Path("data/Kodagu_SAR_GT")
BANDS = ["VV", "VH", "slope"]
PRE_WIN = ("2018-01-15", "2018-07-15")    # before the Aug-2018 storm
POST_WIN = ("2018-09-15", "2019-03-15")   # after


def _composite(ee, region, d0, d1):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]).filterDate(d0, d1))
    return s1.median()


def _download_grid(ee, bounds, out_dir, nx=3, ny=3):
    """Download S1 pre/post over the bbox as an nx*ny grid of GeoTIFFs."""
    slope = ee.Terrain.slope(
        ee.Image("NASA/NASADEM_HGT/001").select("elevation")).rename("slope")
    xmin, ymin, xmax, ymax = bounds
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    pre_tiles, post_tiles = [], []
    for j in range(ny):
        for i in range(nx):
            sub = [xmin + i*dx, ymin + j*dy, xmin + (i+1)*dx, ymin + (j+1)*dy]
            region = ee.Geometry.Rectangle(sub)
            for tag, win, store in (("PRE", PRE_WIN, pre_tiles),
                                    ("POST", POST_WIN, post_tiles)):
                img = _composite(ee, region, *win).addBands(slope).clip(region)
                p = out_dir / f"_{tag}_{j}_{i}.tif"
                for a in range(3):
                    try:
                        url = img.getDownloadURL({"region": region, "scale": 10,
                                                  "format": "GEO_TIFF", "bands": BANDS})
                        urllib.request.urlretrieve(url, p)
                        store.append(p); break
                    except Exception as e:
                        if a == 2:
                            print(f"  [fail] {tag} {j},{i}: {str(e)[:80]}")
                        else:
                            import time; time.sleep(2 * (a + 1))
            print(f"  tile {j},{i} done")
    return pre_tiles, post_tiles


def _mosaic(tiles, dst):
    srcs = [rasterio.open(t) for t in tiles]
    arr, tfm = merge(srcs)
    prof = srcs[0].profile.copy()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tfm,
                count=arr.shape[0], compress="deflate")
    with rasterio.open(dst, "w", **prof) as d:
        d.write(arr)
    for s in srcs:
        s.close()
    return dst


def main():
    import ee
    ee.Initialize(project="major-project-512310")
    OUT.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd
    inv = gpd.read_file(INV).to_crs(4326)
    b = inv.total_bounds
    pad = 0.01
    bounds = [b[0]-pad, b[1]-pad, b[2]+pad, b[3]+pad]
    print(f"Inventory: {len(inv)} polygons, bbox {[round(x,3) for x in bounds]}")

    print("Downloading Sentinel-1 pre/post grid (SAR, cloud-free)...")
    pre_t, post_t = _download_grid(ee, bounds, OUT)
    print(f"Mosaicking {len(pre_t)} pre / {len(post_t)} post tiles...")
    _mosaic(pre_t, OUT / "PRE.tif")
    _mosaic(post_t, OUT / "POST.tif")

    # Rasterise the real inventory polygons onto the PRE grid.
    with rasterio.open(OUT / "PRE.tif") as ref:
        prof, tfm, crs = ref.profile, ref.transform, ref.crs
        H, W = ref.height, ref.width
    inv_p = inv.to_crs(crs)
    mask = rasterize([(g, 1) for g in inv_p.geometry if g is not None],
                     out_shape=(H, W), transform=tfm, fill=0, dtype="uint8")
    mprof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                 crs=crs, transform=tfm, compress="deflate")
    with rasterio.open(OUT / "mask.tif", "w", **mprof) as d:
        d.write(mask, 1)
    for t in pre_t + post_t:
        t.unlink(missing_ok=True)  # drop scratch tiles
    print(f"Real GT mask: {mask.sum()} landslide px ({mask.mean()*100:.3f}%), "
          f"scene {H}x{W} -> {OUT}/")


if __name__ == "__main__":
    main()
