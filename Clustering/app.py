"""
app.py
======
Flask viewer: simulate a flood over a real city map, then identify and rank the
affected zones with the clustering / scoring / tiering in flood_zones.py.

  cities.py      -> where each city floods + a "stage" dial to control how bad it is
  sar_scenes.py  -> real floods: the trained model's output raster as flood marks
  flood_zones.py -> DBSCAN footprints -> severity score (depth x density) -> tiers
  this file      -> serves the map (Leaflet) and the /api/flood endpoint

Cities: New York City (hurricane surge), Chennai (monsoon deluge),
        Cuttack (Mahanadi-delta river flood).

Run:   python app.py      then open  http://127.0.0.1:5001
"""

from __future__ import annotations
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from cities import (CITIES, DEFAULT_CITY, inundated_marks, generate_candidates,
                    city_meta)
from flood_zones import FloodSeverityZones, _hull_ring_lonlat
from routing import compute_routes
from sar_scenes import SCENE_DIR, scene_config, scene_keys, scene_marks, scene_meta

app = Flask(__name__)

TIER_COLOUR = {
    "Critical": "#7f0000",
    "High":     "#d7301f",
    "Moderate": "#fc8d59",
    "Low":      "#fdcc8a",
}


def _safe(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def run_flood(city_key, stage_m, seed, eps_m, min_samples, depth_weight):
    if city_key in CITIES:
        marks = inundated_marks(city_key, stage_m, seed=seed)
        total_candidates = len(generate_candidates(city_key, seed))
        water, meta, config = marks, city_meta(city_key), CITIES[city_key]
    else:
        # A model scene: the water field for routing is every mark the model
        # flooded; the zones are built from the ones within reach of a road.
        water = scene_marks(city_key)
        marks = water[water.exposed].reset_index(drop=True)
        total_candidates = len(water)
        meta, config = scene_meta(city_key), scene_config(city_key)

    result = {
        "city": meta,
        "params": {"city": city_key, "stage_m": stage_m, "seed": seed,
                   "eps_m": eps_m, "min_samples": min_samples,
                   "depth_weight": depth_weight},
        "tier_colour": TIER_COLOUR,
        "marks": {"type": "FeatureCollection", "features": []},
        "zones": {"type": "FeatureCollection", "features": []},
        "routes": {"type": "FeatureCollection", "features": []},
        "summary": [],
        "stats": {"n_wet": int(len(marks)), "n_candidates": int(total_candidates),
                  "n_zones": 0, "n_noise": 0,
                  "max_depth_cm": float(marks.depth_cm.max()) if len(marks) else 0.0},
    }

    if len(marks) < max(min_samples, 3):
        return result

    model = FloodSeverityZones(
        eps_m=eps_m, min_samples=min_samples,
        severity_col="depth_cm", depth_weight=depth_weight,
    )
    model.fit(marks)
    pts = model.points_

    for _, r in pts.iterrows():
        result["marks"]["features"].append({
            "type": "Feature",
            "properties": {
                "depth_cm": _safe(round(float(r["depth_cm"]), 1)),
                "ground_elev_m": _safe(float(r["ground_elev_m"])),
                "area": r["area"], "locality": r["locality"],
                "zone": int(r["zone"]), "noise": bool(r["zone"] == -1),
            },
            "geometry": {"type": "Point",
                         "coordinates": [float(r["lon"]), float(r["lat"])]},
        })

    for _, zr in model.zones_.iterrows():
        g = pts[pts.zone == zr["zone"]]
        ring = _hull_ring_lonlat(g.lon.to_numpy(), g.lat.to_numpy())
        props = {k: _safe(v) for k, v in zr.items()}
        props["fill"] = TIER_COLOUR.get(zr["tier"], "#fdcc8a")
        props["name"] = g["area"].mode().iat[0] if len(g) else "zone"
        props["localities"] = sorted(set(g["locality"]) - {"-"})
        result["zones"]["features"].append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    # --- evacuation routes: every zone -> nearest safe ground outside the city --
    zone_dicts = [
        {"zone": int(zf["properties"]["zone"]), "tier": zf["properties"]["tier"],
         "fill": zf["properties"]["fill"],
         "centroid_lon": zf["properties"]["centroid_lon"],
         "centroid_lat": zf["properties"]["centroid_lat"]}
        for zf in result["zones"]["features"]
    ]
    route_feats, route_info = compute_routes(
        water[["lon", "lat", "depth_cm"]], zone_dicts, config)
    result["routes"]["features"] = route_feats

    result["summary"] = [
        {**{k: _safe(v) for k, v in row.items()},
         "name": zf["properties"]["name"],
         "route": route_info.get(int(zf["properties"]["zone"]))}
        for row, zf in zip(model.zones_.to_dict("records"),
                           result["zones"]["features"])
    ]
    result["stats"]["n_zones"] = int(len(model.zones_))
    result["stats"]["n_noise"] = int(model.n_noise_)
    result["stats"]["n_cutoff"] = sum(
        1 for v in route_info.values() if not v["reachable"])
    return result


@app.route("/")
def index():
    cities = [{"key": k, "name": c["name"]} for k, c in CITIES.items()]
    scenes = [{"key": k, "name": scene_meta(k)["name"]} for k in scene_keys()]
    start = request.args.get("city")
    if start not in CITIES and start not in scene_keys():
        start = DEFAULT_CITY
    return render_template("index.html", cities=cities, scenes=scenes, default_city=start)


@app.route("/scenes/<key>.png")
def scene_overlay(key):
    return send_from_directory(SCENE_DIR, f"{key}.png")


@app.route("/api/flood")
def api_flood():
    city_key = request.args.get("city", default=DEFAULT_CITY)
    if city_key not in CITIES and city_key not in scene_keys():
        city_key = DEFAULT_CITY
    stage_m = None                      # a model scene has no stage: the flood is measured
    if city_key in CITIES:
        c = CITIES[city_key]
        stage_m = request.args.get("stage_m", default=c["stage_default"], type=float)
        stage_m = min(max(stage_m, 0.1), c["stage_max"] + 1.0)
    seed = request.args.get("seed", default=42, type=int)
    eps_m = request.args.get("eps_m", default=400.0, type=float)
    min_samples = request.args.get("min_samples", default=8, type=int)
    depth_weight = request.args.get("depth_weight", default=0.65, type=float)
    depth_weight = min(max(depth_weight, 0.0), 1.0)
    return jsonify(run_flood(city_key, stage_m, seed, eps_m, min_samples, depth_weight))


if __name__ == "__main__":
    import os
    # 5000 is hijacked by macOS AirPlay Receiver on recent macOS; default to 5001.
    # No reloader: it forks the process, which does not mix well with the large
    # NumPy road graphs held in memory.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5001)),
            threaded=True, use_reloader=False)
