# Flood severity zones & evacuation routing

Simulate a flood over a real city, find the affected areas, rank them by severity,
and route every affected zone to safe high ground along the real street network.

Built around `flood_zones.py` (unchanged) with a Flask + Leaflet viewer on top.

```
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:5001   (5000 is taken by macOS AirPlay)
# PORT=8000 python app.py     # to pick another port
```

Open the URL and drag the **Flood stage** slider.

---

## The pipeline

```
        stage (m)                 wet marks              footprints            per zone
 city  ──────────►  flood marks ──────────►  DBSCAN  ──────────►  score = depth×density  ──►  Critical / High
 config            depth = stage−elev       (haversine)          + quantile tiering            Moderate / Low
                                                 │
                                                 ▼
                                    road graph (OpenStreetMap)  ──►  multi-source Dijkstra
                                    edge cost = length ×(1+water)     from the safe points
                                                 │
                                                 ▼
                                    one road route per zone  ──►  nearest safe high ground
```

1. **Flood marks** — each city config lists the neighbourhoods that actually flood,
   each with a synthetic *ground elevation*. A flood is one number, the **stage**
   in metres; a mark is wet when `depth_cm = (stage − ground_elev) × 100 > 0`.
   Raise the stage and the water spreads inland.
2. **Zones** — `flood_zones.py` runs DBSCAN (haversine metric, so `eps` is real
   metres) over the wet marks to get contiguous flooded footprints, discarding
   stray marks as noise.
3. **Severity** — each zone gets a score blending normalised water **depth**
   (mean + p90) and mark **density** (marks/km²), then quantile cuts assign
   **Critical / High / Moderate / Low**.
4. **Evacuation routes** — `routing.py` turns the wet marks into a water-depth
   estimate at every road-network node, weights each road edge by
   `length × (1 + 0.09 · depth_cm)` (impassable over ~2.5 m), and runs one
   multi-source Dijkstra from all of the city's safe points. Each zone follows
   the resulting gradient out to the nearest safe high ground — a real
   turn-by-turn road route that steers around deep water and crosses rivers only
   where OpenStreetMap has a bridge.

---

## Real floods from the trained model

`sar_scenes.py` feeds the output of the Sentinel-1 model in `../flood-management`
into the same zoning and routing. The hand-off is the model's GeoTIFF:

```
# 1. run the model (in ../flood-management) -> predictions/<id>_disaster_map.tif
python predict_cnn.py --model best_model.pth --input-s1 scene.tif      # [--bbox W S E N]
#    any size works: it reads, predicts and writes in 2048 px blocks (512 px model tiles)
#    a raw .SAFE scene: test_custom_s1.py, then geocode.py (ingest needs a geocoded raster)

# 2. turn that raster into a scene (here)
python sar_scenes.py ingest ../flood-management/predictions/<id>_disaster_map.tif \
    --name "Assam, July 2024"            # [--bbox W S E N] [--dem dem.tif] [--safe "Name,lat,lon"]

# 3. restart app.py - the scene is in the City dropdown under "Detected by the flood model"
python sar_scenes.py list
```

What `ingest` does with the raster (class 1 = flood, 2 = permanent water; or a
`*_flood_prob.tif` with `--threshold`):

1. **Area** - a raster wider than 40 km is cropped to the 30 km window holding
   the most flood, unless `--bbox` says otherwise.
2. **Flood marks** - one mark per ~150 m ground cell that is at least 35 % flooded.
3. **Depth** - the model says *where* the water is, not how deep. Depth is
   estimated from a DEM (AWS terrain tiles, or `--dem`): the water surface over a
   pixel is the ground height at the nearest dry edge, minus the ground under it.
   With no DEM it falls back to a distance-from-dry-land proxy.
4. **Roads** - the OpenStreetMap graph for the area, cached like a city's.
5. **Exposed marks** - marks within 500 m of a road. Only these are clustered into
   zones, so priority goes to flooded places people can be routed out of rather
   than to open floodplain or the river channel. Routing still uses *every*
   mark as its water field.
6. **Safe points** - picked automatically: road junctions the model left dry,
   over 600 m from any water, on the higher half of the dry ground, placed where
   they serve the most exposed marks. Pass `--safe` to use real shelters instead.

The viewer hides the stage slider and seed for a scene (the flood is measured,
not simulated) and draws the model's flood extent under the zones.

Caveats: depth from a ~30 m DEM is rough in flat floodplain; without a
pre-flood scene or permanent-water mask the river itself is counted as flood;
auto safe points are dry high ground, not verified shelters.

Checked on the 68 Sen1Floods11 hand-labelled Assam chips, mosaicked into one
17 409 x 17 920 px raster (`best_model.pth`, ResNet34-UNet, 3 min on Apple MPS):
the disaster map scores IoU 0.52 against the hand labels (0.68 on the 14 test-split
chips the model never trained on); ingest gives 52 km² of flood in the 30 km
window around Silghat / Kaliabor, 7 priority zones and 6 routes - the 7th zone,
on the Brahmaputra bank, is 2.6 m deep on average and correctly reported cut off.

The venv also runs the model if you add `torch torchvision matplotlib`
(install with `.venv/bin/python -m pip`, the venv's `pip` script still points at
the folder it was created in).

---

## Cities

Pick from the **City** dropdown, or open `/?city=<key>`:

| key | city | flood driver | modelled flood-prone areas | safe high ground |
|-----|------|--------------|----------------------------|------------------|
| `nyc` | New York City | Atlantic hurricane storm surge (Sandy / FEMA zones) | Battery Park, Red Hook, Gowanus, DUMBO, Lower East Side, Newtown Creek, Hunts Point, LaGuardia, Howard Beach, Broad Channel, the Rockaways, Coney Island, Sheepshead Bay, Staten Island east shore | Central Park, Prospect Park, Forest Park, Todt Hill, Van Cortlandt Park |
| `chennai` | Chennai | NE-monsoon deluge + Chembarambakkam release into the Adyar (Dec 2015) | Velachery, Pallikaranai marsh, Perungudi, Madipakkam, Mudichur, West Tambaram, Saidapet, Kotturpuram, T. Nagar, Mylapore, Cooum banks, Ambattur, Manali, Ennore, Semmancheri, Guindy | St. Thomas Mount, Pallavaram, Anna Nagar, Red Hills ridge, Tambaram |
| `cuttack` | Cuttack, Odisha | Mahanadi-delta river flood (city wedged between the Mahanadi & Kathajodi) | Naraj, Jobra, Sikharpur, Choudhury Bazar, Badambadi, Matagajpur, Tarapur, Bidanasi/CDA, Gandarpur, Trisulia, Jagatpur, Raghunathpur | across the bridges — Bhubaneswar-side high ground (S), Barang uplands (SW), Choudwar ridge (N), Salipur side (E) |

Neighbourhood names and coordinates are real, from news coverage and flood
studies of past events. Elevations, mark counts, safe-point placement and the
stage→depth mapping are **synthetic**, tuned for a legible demo — this is not a
hydrological model.

---

## Controls

| control | effect |
|---------|--------|
| **Flood stage** | The main dial. Label and units change per city (surge height / peak water level / river level above danger mark). The line under the slider snaps to the nearest real event. |
| **DBSCAN radius** | `eps` in metres — how close two marks must be to join the same footprint. Larger merges zones. |
| **Min marks / zone** | DBSCAN `min_samples` — higher means only well-attested zones survive; the rest become "isolated" noise. |
| **Depth weight vs density** | `1.0` ranks zones purely by how deep the water is, `0.0` purely by how concentrated the marks are. |
| **Scenario seed** | A fresh random realisation of the same generative model. |
| **Layers** | Toggle severity zones, evacuation routes, flood marks, noise marks. |

**Zone table** — ranked by severity score. Columns: tier, zone name, score, mark
count, max depth, and **Evac** (bearing + road distance; hover for the
destination name). A zone whose streets are all sealed by > ~2.5 m water shows
**cut off** in red — try Cuttack near max stage, where the island strands.
Hover a row to highlight its zone + route; click to fly along the route.

---

## Project layout

| file | role |
|------|------|
| `flood_zones.py` | The base algorithm, unchanged. `FloodSeverityZones.fit(df)` → `.points_` (labelled marks), `.zones_` (scored + tiered table), `.to_geojson()`. |
| `cities.py` | Per-city config: map view, flood-prone `areas` tuples, real-event `references`, and high-ground `safe_points`. `inundated_marks(city, stage_m, seed)` returns the wet marks; `city_meta(city)` is the front-end description. |
| `roads.py` | Real road network per city. Reads an OpenStreetMap dump from `cache/roads_<city>.json.gz` (fetched from Overpass on first use if absent), parses it into a compact undirected CSR graph cached as `cache/graph_<city>.npz`, and snaps arbitrary lat/lon onto the nearest road node. |
| `sar_scenes.py` | Bridge from the trained flood model. `ingest` turns a model GeoTIFF into a scene (`scenes/<key>.json.gz` + flood-extent PNG): flood marks with DEM depth, road graph, exposed marks, safe points. `scene_marks` / `scene_config` / `scene_meta` are the scene counterparts of `inundated_marks` / the city dict / `city_meta`. |
| `routing.py` | Evacuation routing on that graph — water-depth-weighted multi-source Dijkstra (`scipy.sparse.csgraph`) from the safe points, returning one road route per zone plus distance / deepest-water / cut-off info. |
| `app.py` | Flask. `GET /api/flood?city=&stage_m=&eps_m=&min_samples=&depth_weight=&seed=` runs the whole pipeline; returns city meta + marks + zone polygons + routes (GeoJSON) + ranked summary + stats. `/` serves the page. |
| `templates/index.html`, `static/` | Leaflet viewer. Leaflet is vendored in `static/vendor/` (no CDN for the library); the basemap is OpenStreetMap raster tiles, CSS-darkened. Marks coloured by depth, zones tier-coloured (Critical pulses), routes as marching-ant lines to green safe-point pins. |
| `cache/` | Pre-built road graphs (~54 MB) + gzipped raw Overpass dumps. |
| `requirements.txt` | numpy, pandas, scikit-learn, scipy, flask; rasterio for `sar_scenes.py ingest`. |

---

## Notes & caveats

- **Basemap tiles** need internet at runtime (OpenStreetMap). The Leaflet library
  itself is vendored and works offline.
- **Road graphs** are pre-built in `cache/`. Delete them and the first request
  for that city re-fetches ~50–70 MB from Overpass (~1–2 min) and re-parses —
  one-off, then cached. First request per city after a restart loads its graph
  (~1 s for NYC/Chennai); after that, responses are ~0.1–0.6 s.
- The server runs **without Flask's auto-reloader** (it forks, which clashes with
  the in-memory road graphs) — restart it by hand after editing the Python.
- `flood_zones.py` tiers by score **quantile**, so a fixed fraction of zones
  always lands in each tier regardless of absolute severity. For absolute
  depth/score bands, edit `_score_and_tier`.
- The reported "deepest water on route" includes the flooded zone centroid the
  route starts from, so for a badly flooded zone it will be large by definition.

---

## Adding another city

1. Append a dict to `cities.py` (copy an existing one): `center` / `zoom`,
   `driver` / `stage_label` / `stage_min|max|default`, `areas` tuples
   `(name, locality, lat, lon, n_pts, spread_deg, base_elev_m,
   elev_gain_m_per_km, elev_noise_m)`, a few `references`, and `safe_points`
   `[(name, lat, lon), ...]` on real high ground.
2. Add its Overpass bounding box to `FETCH_BBOX` in `roads.py`, then run
   `python roads.py` once to build and cache the road graph.

It appears in the dropdown automatically.
