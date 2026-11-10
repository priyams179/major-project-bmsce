"""Download historical **Landsat 5** (TM, Collection-2 L2) pre/post imagery for
Indian landslide events from the NASA Global Landslide Catalog, with cloud
cover dropped (scene-level CLOUD_COVER filter + per-pixel QA_PIXEL masking).

Scope note (honest): Landsat 5 operated 1984-2013; Landsat 6 never reached orbit
(no data). The NASA catalog's Indian events start in 2007, so the usable
Landsat-5 window is ~2007-2012. These events have point locations only (no pixel
masks), so training downstream uses change-detection **pseudo-labels** and
accuracy is validated by detection-vs-catalog proximity + visual QC.

Bands downloaded: SR_B4 (NIR), SR_B5 (SWIR1), SR_B7 (SWIR2) -- the change
-detection bands, scaled to surface reflectance. Output: one GeoTIFF pair per
event in data/Landsat5_India/{id}_PRE.tif / _POST.tif + manifest.csv.

Usage:
  python download_landsat5_india.py --n 120 --out data/Landsat5_India
  python download_landsat5_india.py --n 120 --project <your-ee-project>
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from pathlib import Path

GLC = Path("data/GLC_export.csv")
BANDS = ["SR_B4", "SR_B5", "SR_B7"]  # NIR, SWIR1, SWIR2


def _year(r):
    m = re.search(r"(19|20)\d\d", str(r.get("event_date", "")))
    return int(m.group(0)) if m else 0


def _month(r):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(r.get("event_date", "")))
    return int(m.group(1)) if m else 7


# Named region bounding boxes (lat_min, lat_max, lon_min, lon_max).
REGIONS = {
    "westernghats": (8.0, 16.5, 73.5, 78.0),   # Kerala/Karnataka/Goa/Nilgiris
    "himalaya": (28.0, 35.0, 74.0, 81.0),
    "northeast": (22.0, 29.0, 88.0, 97.0),
}


def select_events(n: int, y0=2007, y1=2012, snap=0.1, region=None):
    """Indian medium/large events in the L5 window, spatially de-duplicated.
    region: a key of REGIONS to restrict to one landslide belt (else all India).
    """
    box = REGIONS.get(region) if region else None
    rows = list(csv.DictReader(open(GLC)))
    out, seen = [], set()
    for r in rows:
        if r.get("country_name") != "India" or not r.get("latitude"):
            continue
        y = _year(r)
        if not (y0 <= y <= y1):
            continue
        if r.get("landslide_size") not in ("medium", "large", "very_large"):
            continue
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if box and not (box[0] <= lat <= box[1] and box[2] <= lon <= box[3]):
            continue
        key = (round(lat / snap), round(lon / snap), y)
        if key in seen:
            continue
        seen.add(key)
        eid = f"glc{r.get('event_id','')}_{_slug(r.get('location_description',''))}_{y}"
        out.append({"id": eid, "lat": lat, "lon": lon, "year": y,
                    "month": _month(r), "size": r.get("landslide_size")})
        if len(out) >= n:
            break
    return out


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())[:16] or "india"


def _download(ee, ev, out_dir, min_scenes=3, retries=3, cloud_cover=40,
              window_months=11):
    import rasterio  # noqa: F401
    pt = ee.Geometry.Point(ev["lon"], ev["lat"])
    region = pt.buffer(3840).bounds()
    y, m = ev["year"], ev["month"]

    def masked(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4)).Not()  # not cloud/shadow
        sr = img.select(BANDS).multiply(0.0000275).add(-0.2)
        return sr.updateMask(clear)

    def composite(d0, d1):
        c = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
             .filterBounds(region).filterDate(d0, d1)
             .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover)).map(masked))
        return c.median().clip(region), c.size()

    # Wider windows accumulate more cloud-free observations for cloudy regions;
    # per-pixel QA_PIXEL masking still removes clouds, median composites the rest.
    def shift(months):
        tot = (y * 12 + (m - 1)) + months
        return f"{tot // 12}-{tot % 12 + 1:02d}-15"
    pre, npre = composite(shift(-window_months), shift(-1))
    post, npost = composite(shift(1), shift(window_months))
    npre, npost = npre.getInfo(), npost.getInfo()
    if npre < min_scenes or npost < min_scenes:
        return None  # not enough cloud-free coverage

    for name, img in (("PRE", pre), ("POST", post)):
        for a in range(retries):
            try:
                url = img.getDownloadURL({"region": region, "scale": 30,
                                          "format": "GEO_TIFF", "bands": BANDS})
                urllib.request.urlretrieve(url, out_dir / f"{ev['id']}_{name}.tif")
                break
            except Exception as e:
                if a == retries - 1:
                    print(f"    [fail] {ev['id']} {name}: {str(e)[:70]}")
                    return None
                time.sleep(2 * (a + 1))
    return {"pre_scenes": npre, "post_scenes": npost}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="max events to download")
    ap.add_argument("--out", default="data/Landsat5_India")
    ap.add_argument("--project", default="major-project-512310", help="EE project")
    ap.add_argument("--region", choices=list(REGIONS), default=None,
                    help="restrict to one landslide belt (e.g. westernghats)")
    ap.add_argument("--min-scenes", type=int, default=3,
                    help="min cloud-free scenes per composite (lower for cloudy regions)")
    ap.add_argument("--cloud-cover", type=int, default=40,
                    help="max scene CLOUD_COVER %% to admit")
    ap.add_argument("--window-months", type=int, default=11,
                    help="pre/post window length (wider helps cloudy regions)")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "lat", "lon", "year", "month", "size", "pre_scenes", "post_scenes"]

    # Merge with any existing manifest so repeated region-targeted runs ACCUMULATE.
    manifest = {}
    mpath = out / "manifest.csv"
    if mpath.exists():
        manifest = {r["id"]: r for r in csv.DictReader(open(mpath))}
        print(f"Existing manifest: {len(manifest)} events (new ones appended).")

    events = select_events(args.n, region=args.region)
    tag = f" [{args.region}]" if args.region else ""
    print(f"Selected {len(events)} Indian Landsat-5-era events (2007-2012){tag}.")
    added = 0
    for i, ev in enumerate(events, 1):
        if ev["id"] in manifest:
            continue  # already have it
        r = _download(ee, ev, out, min_scenes=args.min_scenes,
                      cloud_cover=args.cloud_cover,
                      window_months=args.window_months)
        if r:
            manifest[ev["id"]] = {**ev, **r}
            added += 1
            print(f"  [{i}/{len(events)}] {ev['id']} ok "
                  f"(pre {r['pre_scenes']} / post {r['post_scenes']} scenes)")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest.values())
    print(f"\nAdded {added} new pairs; manifest now {len(manifest)} events -> {out}")


if __name__ == "__main__":
    main()
