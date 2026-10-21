"""Geocode results: vectorize masks to polygons, attach lat/lon + place names.

Outputs:
  results/landslides.geojson   - polygons in EPSG:4326
  results/landslides.csv       - one row per polygon (centroid, area, place)
  results/event_summary.csv    - one row per event (total area, counts)

Reverse geocoding: offline gazetteer from landslide_events.csv (nearest
known event location). Optional online Nominatim lookup behind --online.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import Point, mapping, shape

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
EVENTS_CSV = Path("landslide_events.csv")


def load_event_points() -> pd.DataFrame:
    rows = []
    with open(EVENTS_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "event_id": r["event_id"].strip(),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "event_date": r["event_date"].strip(),
            })
    return pd.DataFrame(rows)


def vectorize_mask(tif_path: Path, min_px: int = 5, px_size_m: float = 30.0) -> tuple[list[dict], object]:
    """Raster mask -> list of polygon dicts with area. Landsat ~30m pixels."""
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms = []
    # Area per pixel in CRS units.
    px_crs = abs(transform.a) * abs(transform.e)
    for geom, val in features.shapes(mask, mask=mask.astype(bool), transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        n_px = g.area / max(px_crs, 1e-18)
        if n_px < min_px:
            continue
        area_m2 = n_px * (px_size_m**2)
        geoms.append({"geometry": g, "area_m2": area_m2, "px_area": n_px})
    return geoms, crs


def reverse_geocode(lat: float, lon: float, events: pd.DataFrame) -> tuple[str, str]:
    """Nearest known event location as place label (offline)."""
    d2 = (events["lat"] - lat) ** 2 + (events["lon"] - lon) ** 2
    i = d2.idxmin()
    row = events.loc[i]
    # ~km distance
    km = math.sqrt(d2[i]) * 111.0
    return f"near {row['event_id']} (~{km:.0f} km)", str(row["event_id"])


def geocode_results(mask_dir: Path = Path("results/masks_ens"), tag: str = "") -> None:
    events_df = load_event_points()

    all_polys = []
    summary = []

    for tif in sorted(mask_dir.glob("*_mask.tif")):
        eid = tif.name[: -len("_mask.tif")]
        geoms, crs = vectorize_mask(tif)

        ev_row = events_df.loc[events_df["event_id"] == eid]
        if len(ev_row) > 0:
            rec = ev_row.iloc[0]
            ev_lat, ev_lon = float(rec["lat"]), float(rec["lon"])
            ev_date = str(rec["event_date"])
        else:
            ev_lat = ev_lon = float("nan")
            ev_date = ""

        total_km2 = 0.0
        for g in geoms:
            # Pixel area m2 -> km2 (30m pixels approx).
            area_km2 = g["area_m2"] / 1e6
            total_km2 += area_km2
            c = g["geometry"].centroid
            lat, lon = c.y, c.x
            place, _ = reverse_geocode(lat, lon, events_df)
            all_polys.append({
                "event_id": eid,
                "event_date": ev_date,
                "lat": lat,
                "lon": lon,
                "area_km2": round(area_km2, 4),
                "place": place,
                "geometry": g["geometry"],
            })

        summary.append({
            "event_id": eid,
            "event_date": ev_date,
            "event_lat": ev_lat,
            "event_lon": ev_lon,
            "n_polygons": len(geoms),
            "total_area_km2": round(total_km2, 4),
        })
        print(f"{eid}: {len(geoms)} polygons, {total_km2:.3f} km2")

    # GeoJSON (EPSG:4326).
    if all_polys:
        gdf = gpd.GeoDataFrame(all_polys, geometry="geometry", crs="EPSG:4326")
        # Reproject from source CRS if needed.
        with rasterio.open(next(mask_dir.glob("*_mask.tif"))) as src:
            if src.crs and src.crs.to_epsg() != 4326:
                gdf = gdf.set_crs(src.crs).to_crs("EPSG:4326")
        gj_path = RESULTS / f"landslides{tag}.geojson"
        gdf.to_file(gj_path, driver="GeoJSON")
        print(f"Wrote {gj_path} ({len(gdf)} polygons)")

        # CSV without geometry.
        csv_df = gdf.drop(columns="geometry")
        csv_path = RESULTS / f"landslides{tag}.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")
    else:
        print("No polygons found.")

    sum_path = RESULTS / f"event_summary{tag}.csv"
    pd.DataFrame(summary).to_csv(sum_path, index=False)
    print(f"Wrote {sum_path}")
