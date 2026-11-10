"""Download **Sentinel-1 C-band SAR** (GRD, IW mode) pre/post imagery for
**Western Ghats** landslide events — the cloud-penetrating sensor that works
through the monsoon, unlike optical Landsat-5 (which scored 0% there).

Why SAR / which bands (from the landslide-SAR literature, see REPORT §2.5):
  * **VV + VH** backscatter (dB) — VV ~ surface roughness/moisture, VH ~
    vegetation volume scattering (landslides strip vegetation). Use both.
  * Change-detection layers (ΔVV, ΔVH log-ratio) are the most discriminative
    signal — handled downstream in src/sentinel1_wg.py.
  * **slope** (NASADEM) — landslides sit on slopes; used to mask flats.
  * Coherence (InSAR) is more sensitive but needs SLC processing (not in GEE
    GRD) and may not improve detection — omitted.

Output: data/Sentinel1_WG/{id}_PRE.tif / _POST.tif  (bands: VV, VH, slope),
10 m, + manifest.csv. SAR is all-weather, so NO cloud filter is needed;
median over many acquisitions also suppresses speckle.

Usage:
  python download_sentinel1_wg.py --n 40 --out data/Sentinel1_WG
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from pathlib import Path

GLC = Path("data/GLC_export.csv")
BANDS = ["VV", "VH", "slope"]

# Documented Western Ghats landslide events (2018-2024), beyond the GLC window
# (dense Sentinel-1 coverage). Coordinates are event-area centroids (±few km is
# fine: the accuracy metric uses a 3 km radius).
RECENT_WG = [
    # --- Kodagu / Karnataka Western Ghats ---
    ("kodagu_madikeri_2018", 12.42, 75.73, 2018, 8),
    ("kodagu_mukkodlu_2019", 12.35, 75.70, 2019, 8),
    ("shirur_2024", 14.6036, 74.3708, 2024, 7),
    ("ankola_2024", 14.66, 74.30, 2024, 7),
    # --- Kerala Western Ghats ---
    ("keralaidukki_2018", 9.85, 77.16, 2018, 8),
    ("munnar_2018", 10.09, 77.06, 2018, 8),
    ("nilambur_2018", 11.27, 76.23, 2018, 8),
    ("kozhikode_2018", 11.50, 75.90, 2018, 8),
    ("wayanad_2018", 11.60, 76.10, 2018, 8),
    ("kavalappara_2019", 11.41, 76.24, 2019, 8),
    ("puthumala_2019", 11.48, 76.08, 2019, 8),
    ("pettimudi_2020", 10.13, 77.07, 2020, 8),
    ("koottickal_2021", 9.68, 76.86, 2021, 10),
    ("kokkayar_2021", 9.66, 76.90, 2021, 10),
    ("wayanad_2024", 11.4646, 76.1348, 2024, 7),
    # --- Nilgiris / Tamil Nadu Western Ghats ---
    ("coonoor_2019", 11.35, 76.79, 2019, 8),
    ("kotagiri_2019", 11.42, 76.86, 2019, 8),
    ("kodaikanal_2018", 10.24, 77.49, 2018, 8),
    # --- Maharashtra Sahyadri (northern Western Ghats) ---
    ("taliye_2021", 17.73, 73.47, 2021, 7),
    ("irshalwadi_2023", 18.98, 73.22, 2023, 7),
]
WG_BOX = (8.0, 16.5, 73.5, 78.0)  # lat_min, lat_max, lon_min, lon_max


def _year(r):
    m = re.search(r"(19|20)\d\d", str(r.get("event_date", "")))
    return int(m.group(0)) if m else 0


def _month(r):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", str(r.get("event_date", "")))
    return int(m.group(1)) if m else 7


def select_events(n, snap=0.1):
    """GLC Western Ghats events 2015-2017 + the recent major events."""
    out, seen = [], set()
    for eid, lat, lon, y, m in RECENT_WG:
        out.append({"id": f"wg_{eid}", "lat": lat, "lon": lon, "year": y, "month": m})
        seen.add((round(lat / snap), round(lon / snap), y))
    for r in csv.DictReader(open(GLC)):
        if r.get("country_name") != "India" or not r.get("latitude"):
            continue
        y = _year(r)
        if not (2015 <= y <= 2017):
            continue
        lat, lon = float(r["latitude"]), float(r["longitude"])
        if not (WG_BOX[0] <= lat <= WG_BOX[1] and WG_BOX[2] <= lon <= WG_BOX[3]):
            continue
        key = (round(lat / snap), round(lon / snap), y)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"wg_glc{r.get('event_id','')}_{y}",
                    "lat": lat, "lon": lon, "year": y, "month": _month(r)})
        if len(out) >= n:
            break
    return out[:n]


def _download(ee, ev, out_dir, window_months=6, min_scenes=3, retries=3):
    pt = ee.Geometry.Point(ev["lon"], ev["lat"])
    region = pt.buffer(1920).bounds()           # ~384 px at 10 m
    y, m = ev["year"], ev["month"]
    slope = ee.Terrain.slope(ee.Image("NASA/NASADEM_HGT/001").select("elevation")).rename("slope")

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))

    def shift(months):
        tot = (y * 12 + (m - 1)) + months
        return f"{tot // 12}-{tot % 12 + 1:02d}-15"

    def comp(d0, d1):
        c = s1.filterDate(d0, d1)
        return c.median(), c.size()  # median over many SAR passes -> speckle down

    pre, npre = comp(shift(-window_months), shift(-1))
    post, npost = comp(shift(1), shift(window_months))
    npre, npost = npre.getInfo(), npost.getInfo()
    if npre < min_scenes or npost < min_scenes:
        return None

    for name, img in (("PRE", pre), ("POST", post)):
        stack = img.addBands(slope).clip(region)
        for a in range(retries):
            try:
                url = stack.getDownloadURL({"region": region, "scale": 10,
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
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/Sentinel1_WG")
    ap.add_argument("--project", default="major-project-512310")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=args.project)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "lat", "lon", "year", "month", "pre_scenes", "post_scenes"]
    manifest = {}
    if (out / "manifest.csv").exists():
        manifest = {r["id"]: r for r in csv.DictReader(open(out / "manifest.csv"))}

    events = select_events(args.n)
    print(f"Selected {len(events)} Western Ghats events (2015-2024) for Sentinel-1.")
    added = 0
    for i, ev in enumerate(events, 1):
        if ev["id"] in manifest:
            continue
        r = _download(ee, ev, out)
        if r:
            manifest[ev["id"]] = {**ev, **r}
            added += 1
            print(f"  [{i}/{len(events)}] {ev['id']} ok "
                  f"(pre {r['pre_scenes']} / post {r['post_scenes']} SAR scenes)")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest.values())
    print(f"\nAdded {added} new pairs; manifest now {len(manifest)} events -> {out}")


if __name__ == "__main__":
    main()
