"""Fetch real landslide polygons from NASA COOLR / GLC (ArcGIS REST).

Queries COOLR_Events_Polygons and COOLR_Reports_Polygons for features
near each event in landslide_events.csv. Rasterizes hits onto our
Landsat grid and saves as data/real_masks/<event_id>.npy (uint8).

Fallback: if no polygons found, caller should use dNBR pseudo-mask.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import box, mapping, shape

EVENTS_CSV = Path("landslide_events.csv")
DATA_DIR = Path("data/Landslide_Landsat_Global")
OUT_DIR = Path("data/real_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLY_URL = (
    "https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/"
    "COOLR_Events_Polygons/FeatureServer/0/query"
)
REPORT_URL = (
    "https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/"
    "COOLR_Reports_Polygons/FeatureServer/0/query"
)


def load_events():
    rows = []
    with open(EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "event_id": r["event_id"].strip(),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "date": r["event_date"].strip(),
            })
    return rows


def query_coolr(url: str, lat: float, lon: float, radius_km: float = 15.0):
    """ArcGIS query: features intersecting a point buffer, GeoJSON out."""
    # ~degrees buffer
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(abs(__import__("math").cos(__import__("math").radians(lat))), 0.05))
    envelope = {
        "xmin": lon - dlon,
        "ymin": lat - dlat,
        "xmax": lon + dlon,
        "ymax": lat + dlat,
        "spatialReference": {"wkid": 4326},
    }
    params = {
        "geometry": json.dumps(envelope),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": "200",
    }
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "landslide-project/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"    query error: {e}")
        return []

    feats = data.get("features", [])
    return [shape(f["geometry"]) for f in feats if f.get("geometry")]


def rasterize_geoms(geoms, profile) -> np.ndarray:
    """Burn shapely polygons onto Landsat grid (1 = landslide)."""
    h, w = profile["height"], profile["width"]
    transform = profile["transform"]
    mask = np.zeros((h, w), dtype=np.uint8)
    for g in geoms:
        if g.is_empty:
            continue
        m = features.rasterize(
            [(mapping(g), 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        mask = np.maximum(mask, m)
    return mask


def main():
    events = load_events()
    print(f"Fetching COOLR polygons for {len(events)} events...")
    stats = {"events": 0, "with_labels": 0, "total_polys": 0}

    for ev in events:
        eid = ev["event_id"]
        tif = DATA_DIR / f"{eid}_PRE.tif"
        if not tif.exists():
            print(f"  {eid}: no imagery, skip")
            continue

        with rasterio.open(tif) as src:
            profile = src.profile

        geoms = query_coolr(POLY_URL, ev["lat"], ev["lon"], radius_km=20)
        src_name = "COOLR_Events"
        if not geoms:
            geoms = query_coolr(REPORT_URL, ev["lat"], ev["lon"], radius_km=20)
            src_name = "COOLR_Reports"

        # Keep only geoms intersecting the scene bbox (reduce edge noise).
        with rasterio.open(tif) as src:
            b = src.bounds
            scene_box = box(b.left, b.bottom, b.right, b.top)
        geoms = [g for g in geoms if g.intersects(scene_box) and not g.is_empty]

        if not geoms:
            print(f"  {eid}: 0 polygons")
            stats["events"] += 1
            continue

        mask = rasterize_geoms(geoms, profile)
        frac = mask.mean()
        np.save(OUT_DIR / f"{eid}.npy", mask)
        print(f"  {eid}: {len(geoms)} polys ({src_name}), {frac*100:.2f}% cover")
        stats["events"] += 1
        stats["with_labels"] += 1
        stats["total_polys"] += len(geoms)

    print(
        f"\nDone: {stats['with_labels']}/{stats['events']} events "
        f"with real labels, {stats['total_polys']} polygons total"
    )


if __name__ == "__main__":
    main()
