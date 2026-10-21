"""Independent validation: our detections vs Global Landslide Catalog (GLC).

GLC = 11k real documented landslide points (media/EM-DAT/AGU reports).
Caveat: export covers 1988-2017 only -> only our events <= 2017-09 testable.

For each of our events with GLC points nearby (0.2 deg, +-60 d), distance
from each point to nearest polygon of that event. 0 polygons = miss (inf).
Control: distance to polygons of a random other event (chance level).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

GLC_CSV = Path("data/GLC_export.csv")
EVENTS_CSV = Path("results/event_summary.csv")
GEOJSON = Path(
    sys.argv[1] if len(sys.argv) > 1 else "results/landslides.geojson"
)
OUT = Path(
    Path(GEOJSON).stem + "_glc_report.csv"
    if len(sys.argv) > 1
    else "results/qc_glc_report.csv"
)

RADIUS_DEG = 0.2
DATE_TOL_DAYS = 60
MAX_ACCURACY_KM = 10.0
HIT_M = (100, 300, 1000)


def min_dist_m(point_ll, polys: gpd.GeoDataFrame) -> float:
    """Distance point(lon,lat) -> nearest polygon, meters. inf if no polys."""
    if len(polys) == 0:
        return math.inf
    pt = gpd.GeoSeries([Point(point_ll)], crs="EPSG:4326")
    m = polys.to_crs(pt.estimate_utm_crs())
    return float(m.distance(pt.to_crs(m.crs).iloc[0]).min())


def main() -> None:
    glc = pd.read_csv(GLC_CSV, low_memory=False)
    glc = glc.dropna(subset=["latitude", "longitude", "event_date"])
    glc["date"] = pd.to_datetime(
        glc["event_date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    glc = glc.dropna(subset=["date"])
    acc = pd.to_numeric(glc["location_accuracy"], errors="coerce")
    glc = glc[acc.isna() | (acc <= MAX_ACCURACY_KM)]

    events = pd.read_csv(EVENTS_CSV)
    polys = gpd.read_file(GEOJSON)
    if polys.crs is None:
        polys = polys.set_crs("EPSG:4326")
    rng = np.random.RandomState(42)

    print(f"GLC points: {len(glc)} (to {glc['date'].max().date()}) | "
          f"our events: {len(events)} | our polys: {len(polys)}")

    rows = []
    for _, ev in events.iterrows():
        if math.isnan(ev["event_lat"]):
            continue
        evd = pd.to_datetime(ev["event_date"])
        if evd > glc["date"].max():
            continue  # GLC too old for this event
        near = glc[
            ((glc["latitude"] - ev["event_lat"]).abs() <= RADIUS_DEG)
            & ((glc["longitude"] - ev["event_lon"]).abs() <= RADIUS_DEG)
            & ((glc["date"] - evd).dt.days.abs() <= DATE_TOL_DAYS)
        ]
        if near.empty:
            continue

        ev_polys = polys[polys["event_id"] == ev["event_id"]]
        others = polys[polys["event_id"] != ev["event_id"]]

        for _, p in near.iterrows():
            coord = (float(p["longitude"]), float(p["latitude"]))
            d = min_dist_m(coord, ev_polys)
            # control: random other event's polygons (same geometry pool)
            ctrl_id = others["event_id"].unique()
            pick = ctrl_id[rng.randint(len(ctrl_id))] if len(ctrl_id) else None
            d_ctrl = (
                min_dist_m(coord, others[others["event_id"] == pick])
                if pick is not None
                else math.inf
            )
            rows.append({
                "event_id": ev["event_id"],
                "n_our_polys": int(ev["event_id"] and len(ev_polys)),
                "glc_lat": p["latitude"],
                "glc_lon": p["longitude"],
                "glc_title": str(p["event_title"])[:70],
                "glc_trigger": p["landslide_trigger"],
                "dist_to_our_poly_m": round(d, 1),
                "dist_to_random_event_m": round(d_ctrl, 1),
            })

    if not rows:
        print("No GLC matches.")
        return

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT, index=False)

    d = rep["dist_to_our_poly_m"].to_numpy(dtype=float)
    dc = rep["dist_to_random_event_m"].to_numpy(dtype=float)
    print(f"\nMatched GLC points: {len(d)} across "
          f"{rep['event_id'].nunique()} events "
          f"(events with 0 detections count as miss)\n")
    print(f"{'threshold':>10} {'ours':>8} {'chance':>8}")
    for h in HIT_M:
        a = (d <= h).mean()
        b = (dc <= h).mean()
        print(f"{h:>8} m {a * 100:7.1f}% {b * 100:7.1f}%")
    fin = d[np.isfinite(d)]
    print(f"\nmedian: ours {np.median(fin):.0f} m ({len(d)-len(fin)} misses) | "
          f"chance {np.median(dc[np.isfinite(dc)]):.0f} m")
    print(f"report -> {OUT}")


if __name__ == "__main__":
    main()
