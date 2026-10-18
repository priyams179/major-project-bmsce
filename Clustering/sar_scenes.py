"""
sar_scenes.py
=============
Bridge from the trained flood model (../flood-management) to the zoning and
routing in this folder.

The model's result is a GeoTIFF: the `*_disaster_map.tif` written by
predict_cnn.py (or by test_custom_s1.py followed by geocode.py for a raw scene)
with classes 0 dry / 1 flood / 2 permanent water / 255 excluded, or the
`*_flood_prob.tif` probability raster. `ingest()` turns one of those into a
*scene* - everything a city in cities.py provides, but measured instead of
simulated:

    flood pixels   ->  flood marks     one per flooded ground cell (~150 m)
    DEM            ->  depth_cm        water surface at the nearest dry edge
                                       minus the ground under the mark
    OpenStreetMap  ->  road graph      cache/graph_<key>.npz, same as a city
                   ->  exposed marks   flooded cells within reach of a road -
                                       the ones clustered into priority zones
    DEM + roads    ->  safe points     dry road junctions on higher ground, placed
                                       where they serve the most exposed marks

A scene is stored as `scenes/<key>.json.gz` (+ a PNG of the flood extent for the
map). app.py lists scenes next to the cities; flood_zones.py and routing.py run
on them unchanged.

    python sar_scenes.py ingest ../flood-management/predictions/X_disaster_map.tif \
        --name "Assam, July 2024"  [--bbox W S E N]  [--dem dem.tif]
    python sar_scenes.py list

rasterio is only needed for `ingest`; serving a scene needs nothing extra.
"""

from __future__ import annotations
import argparse
import gzip
import json
import math
import os
import re
import struct
import time
import urllib.parse
import urllib.request
import zlib

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import roads

SCENE_DIR = os.path.join(os.path.dirname(__file__), "scenes")
DEM_DIR = os.path.join(roads.CACHE_DIR, "dem")
_DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"
_M_PER_DEG = 111_320.0

MAX_SIDE_KM = 40.0        # a raster wider than this is cropped to an AOI
AUTO_AOI_KM = 30.0        # side of the AOI picked when --bbox is not given
STRIDE_M = 150.0          # ground spacing of the flood marks
MIN_FILL = 0.35           # a cell becomes a mark when this much of it is flooded
MAX_MARKS = 12_000        # the stride is widened until the marks fit under this
EXPOSURE_M = 500.0        # a mark this close to a road counts as exposed
MIN_DEPTH_CM = 10.0       # the model saw water, so never report less than this
MAX_DEPTH_CM = 600.0
PROXY_CM_PER_M = 0.5      # no-DEM fallback: depth grows with distance from dry land
SAFE_BUFFER_M = 600.0     # a safe point keeps at least this far from any water
SAFE_ELEV_Q = 0.5         # ...and sits above this quantile of the dry junctions
SAFE_REACH_M = 3000.0     # a safe point serves the exposed marks within this range
N_SAFE = 8
PLACE_RADIUS_M = 4000.0   # a mark is named after a place no farther than this


# --------------------------------------------------------------------------- #
# 1. the model raster -> flood / permanent-water masks on a lat/lon grid       #
# --------------------------------------------------------------------------- #
def _classes(arr, threshold):
    """(flood, permanent water) boolean masks from a class or probability raster."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.nan_to_num(arr, nan=0.0) > threshold, np.zeros(arr.shape, bool)
    return arr == 1, arr == 2


def _auto_aoi(vrt, threshold):
    """The AUTO_AOI_KM square holding the most flood, as (w, s, e, n)."""
    f = max(1, int(round(500.0 / (abs(vrt.transform.e) * _M_PER_DEG))))   # ~500 m cells
    h, w = max(1, vrt.height // f), max(1, vrt.width // f)
    flood, _ = _classes(vrt.read(1, out_shape=(h, w)), threshold)
    lat_c = vrt.bounds.bottom + (vrt.bounds.top - vrt.bounds.bottom) / 2
    cell_lat = (vrt.bounds.top - vrt.bounds.bottom) / h
    cell_lon = (vrt.bounds.right - vrt.bounds.left) / w
    ky = max(1, int(AUTO_AOI_KM * 1000 / (cell_lat * _M_PER_DEG)))
    kx = max(1, int(AUTO_AOI_KM * 1000 / (cell_lon * _M_PER_DEG * math.cos(math.radians(lat_c)))))
    dens = ndimage.uniform_filter(flood.astype(np.float32), size=(ky, kx), mode="constant")
    iy, ix = np.unravel_index(int(dens.argmax()), dens.shape)
    y0 = min(max(iy - ky // 2, 0), max(h - ky, 0))
    x0 = min(max(ix - kx // 2, 0), max(w - kx, 0))
    return (vrt.bounds.left + x0 * cell_lon, vrt.bounds.top - (y0 + ky) * cell_lat,
            vrt.bounds.left + (x0 + kx) * cell_lon, vrt.bounds.top - y0 * cell_lat)


def _read_masks(path, bbox, threshold):
    """-> flood, perm (bool HxW), affine transform (EPSG:4326), bbox (w, s, e, n)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise SystemExit(
                f"{path} has no CRS. A raw-scene map from test_custom_s1.py is still "
                "in radar geometry - run flood-management/geocode.py on it first.")
        with WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.nearest) as vrt:
            b = vrt.bounds
            lat_c = (b.bottom + b.top) / 2
            wide = max((b.top - b.bottom) * _M_PER_DEG,
                       (b.right - b.left) * _M_PER_DEG * math.cos(math.radians(lat_c))) / 1000
            if bbox is None:
                bbox = (b.left, b.bottom, b.right, b.top)
                if wide > MAX_SIDE_KM:
                    bbox = _auto_aoi(vrt, threshold)
                    print(f"  raster spans {wide:.0f} km - using the {AUTO_AOI_KM:.0f} km "
                          "window with the most flood (override with --bbox W S E N)")
            w, s, e, n = (max(bbox[0], b.left), max(bbox[1], b.bottom),
                          min(bbox[2], b.right), min(bbox[3], b.top))
            if w >= e or s >= n:
                raise SystemExit("--bbox does not overlap the raster "
                                 f"({b.left:.4f} {b.bottom:.4f} {b.right:.4f} {b.top:.4f})")
            win = from_bounds(w, s, e, n, vrt.transform).round_offsets().round_lengths()
            arr = vrt.read(1, window=win)
            tr = vrt.window_transform(win)
    flood, perm = _classes(arr, threshold)
    h, wd = arr.shape
    return flood, perm, tr, (tr.c, tr.f + h * tr.e, tr.c + wd * tr.a, tr.f)


# --------------------------------------------------------------------------- #
# 2. elevation on the same grid                                               #
# --------------------------------------------------------------------------- #
def _tile(lat, lon, z):
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _dem_tiles(bbox):
    """AWS terrain tiles covering bbox (same source as flood-management/postprocess.py)."""
    w, s, e, n = bbox
    for z in (12, 11, 10, 9):
        x0, y0 = _tile(n, w, z)
        x1, y1 = _tile(s, e, z)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= 16:
            break
    os.makedirs(DEM_DIR, exist_ok=True)
    paths = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            p = os.path.join(DEM_DIR, f"dem_{z}_{x}_{y}.tif")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                req = urllib.request.Request(_DEM_URL.format(z=z, x=x, y=y),
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r, open(p + ".tmp", "wb") as f:
                    f.write(r.read())
                os.replace(p + ".tmp", p)
            paths.append(p)
    return paths


def _dem_on_grid(bbox, shape, tr, dem_path=None):
    """Elevation in metres resampled onto the mask grid, NaN where unknown; None if none."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    try:
        srcs = [dem_path] if dem_path else _dem_tiles(bbox)
    except Exception as exc:            # noqa: BLE001
        print(f"  DEM fetch failed ({exc})")
        return None
    dem = np.full(shape, np.nan, np.float32)
    for p in srcs:
        part = np.full(shape, np.nan, np.float32)
        with rasterio.open(p) as src:
            reproject(rasterio.band(src, 1), part, dst_transform=tr, dst_crs="EPSG:4326",
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        dem = np.where(np.isnan(dem), part, dem)
    dem[dem < -500] = np.nan
    return None if np.isnan(dem).all() else dem


# --------------------------------------------------------------------------- #
# 3. water depth                                                              #
# --------------------------------------------------------------------------- #
def _depth_cm(flood, perm, dem, px_m):
    """
    Depth of the flood water in cm at every flooded pixel.

    With a DEM: the water surface over a pixel is taken as the ground height of
    the nearest dry pixel (where the water's edge meets the land), and depth is
    that surface minus the ground below - the FwDET idea. Without one, depth
    falls back to a proxy that grows with distance from the dry edge.
    """
    dist, (iy, ix) = ndimage.distance_transform_edt(flood | perm, return_indices=True)
    proxy = np.clip(dist * px_m * PROXY_CM_PER_M, MIN_DEPTH_CM, 300.0)
    if dem is None:
        return proxy.astype(np.float32), "edge-distance proxy (no DEM)"
    known = ~np.isnan(dem)
    ground = ndimage.gaussian_filter(np.where(known, dem, np.nanmedian(dem)),
                                     sigma=max(1.0, 30.0 / px_m))
    depth = (ground[iy, ix] - ground) * 100.0
    depth = np.where(known & known[iy, ix], depth, proxy)
    return (np.clip(depth, MIN_DEPTH_CM, MAX_DEPTH_CM).astype(np.float32),
            "DEM: nearest dry-edge elevation minus ground")


# --------------------------------------------------------------------------- #
# 4. pixels -> marks                                                          #
# --------------------------------------------------------------------------- #
def _marks(flood, perm, depth, dem, tr, px_m, stride_m):
    """One mark per mostly-flooded ground cell. Also returns every wet cell centre
    (flood or permanent water) as an (N,2) lon/lat array, and the stride used."""
    h, w = flood.shape
    k = max(1, int(round(stride_m / px_m)))
    while True:
        h2, w2 = h // k * k, w // k * k
        blk = lambda a: a[:h2, :w2].reshape(h2 // k, k, w2 // k, k)     # noqa: E731
        n_fl = blk(flood).sum(axis=(1, 3))
        keep = n_fl >= MIN_FILL * k * k
        if keep.sum() <= MAX_MARKS:
            break
        k += max(1, k // 4)
    wet = (n_fl + blk(perm).sum(axis=(1, 3))) >= MIN_FILL * k * k
    centre = lambda by, bx: np.column_stack([tr.c + (bx * k + k / 2) * tr.a,   # noqa: E731
                                             tr.f + (by * k + k / 2) * tr.e])
    by, bx = np.nonzero(keep)
    ll = centre(by, bx)
    d = blk(np.where(flood, depth, 0.0)).sum(axis=(1, 3))[keep] / n_fl[keep]
    df = pd.DataFrame({"lon": ll[:, 0].round(6), "lat": ll[:, 1].round(6),
                       "depth_cm": d.round(1)})
    if dem is not None:
        ok = ~np.isnan(blk(dem))
        cnt = ok.sum(axis=(1, 3))[keep]
        tot = np.where(ok, blk(dem), 0.0).sum(axis=(1, 3))[keep]
        df["ground_elev_m"] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).round(1)
    else:
        df["ground_elev_m"] = np.nan
    return df, centre(*np.nonzero(wet)), k * px_m


# --------------------------------------------------------------------------- #
# 5. names, safe points, overlay                                              #
# --------------------------------------------------------------------------- #
def _fetch_places(bbox):
    """Named OSM settlements inside bbox as [(name, lat, lon)]; [] if unreachable."""
    w, s, e, n = bbox
    q = ('[out:json][timeout:60];node["place"~"^(city|town|suburb|village|hamlet|'
         f'neighbourhood)$"]["name"]({s},{w},{n},{e});out;')
    body = urllib.parse.urlencode({"data": q}).encode()
    for host in roads._MIRRORS * 2:     # a mirror often throttles right after the road fetch
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(host, data=body, headers=roads._UA), timeout=60) as r:
                els = json.load(r)["elements"]
            return [(el["tags"].get("name:en") or el["tags"]["name"], el["lat"], el["lon"])
                    for el in els]
        except Exception:               # noqa: BLE001
            time.sleep(5)
    return []


def _sector(lon, lat, lon0, lat0):
    brg = np.degrees(np.arctan2(lon - lon0, lat - lat0)) % 360
    names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return names[((brg + 22.5) // 45).astype(int) % 8]


def _name_points(lon, lat, places, bbox):
    """Nearest settlement name for each point, or a compass sector of the scene."""
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    out = np.char.add(_sector(np.asarray(lon), np.asarray(lat), lon0, lat0), " sector")
    if places:
        mx = _M_PER_DEG * math.cos(math.radians(lat0))
        tree = cKDTree([[p[2] * mx, p[1] * _M_PER_DEG] for p in places])
        dist, idx = tree.query(np.column_stack([np.asarray(lon) * mx,
                                                np.asarray(lat) * _M_PER_DEG]))
        near = np.array([p[0] for p in places], dtype=object)[idx]
        out = np.where(dist <= PLACE_RADIUS_M, near, out.astype(object))
    return out


def _safe_points(g, wet_ll, demand_ll, dem, tr, places, bbox):
    """
    Where to evacuate to: [(name, lat, lon)].

    Candidates are road junctions on the main road network that the model left
    dry, well clear of any water and on the higher half of the dry ground. From
    those, pick greedily the ones within SAFE_REACH_M of the most exposed marks
    not yet served - so safe points end up next to the flooding, not just at the
    edge of the scene.
    """
    lat, lon = g.coords[:, 0].astype(float), g.coords[:, 1].astype(float)
    mx = _M_PER_DEG * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    to_xy = lambda ll: np.column_stack([ll[:, 0] * mx, ll[:, 1] * _M_PER_DEG])    # noqa: E731
    xy = np.column_stack([lon * mx, lat * _M_PER_DEG])

    adj = csr_matrix((np.ones(len(g.indices), np.int8), g.indices, g.indptr),
                     shape=(g.n_nodes, g.n_nodes))
    _, comp = connected_components(adj, directed=False)
    main = comp == np.bincount(comp).argmax()
    inside = (lon > bbox[0]) & (lon < bbox[2]) & (lat > bbox[1]) & (lat < bbox[3])
    d_wet = (cKDTree(to_xy(wet_ll)).query(xy)[0] if len(wet_ll)
             else np.full(g.n_nodes, np.inf))

    elev = np.full(g.n_nodes, np.nan)
    if dem is not None:
        col = np.clip(((lon - tr.c) / tr.a).astype(int), 0, dem.shape[1] - 1)
        row = np.clip(((lat - tr.f) / tr.e).astype(int), 0, dem.shape[0] - 1)
        elev = dem[row, col].astype(float)

    degree = np.diff(g.indptr)
    for buffer_m, min_deg in ((SAFE_BUFFER_M, 3), (SAFE_BUFFER_M / 2, 3), (SAFE_BUFFER_M / 4, 2)):
        cand = main & inside & (d_wet > buffer_m) & (degree >= min_deg)
        if cand.sum() >= N_SAFE:
            break
    if not cand.any():
        return []
    if np.isfinite(elev[cand]).any():
        high = cand & (elev >= np.nanquantile(elev[cand], SAFE_ELEV_Q))
        cand = high if high.sum() >= N_SAFE else cand
    ids = np.nonzero(cand)[0]
    tie = np.nan_to_num(elev[ids], nan=0.0)
    tie = tie / (np.abs(tie).max() + 1.0)               # < 1: only breaks ties, upward

    chosen = []
    if len(demand_ll):
        reach = cKDTree(xy[ids]).sparse_distance_matrix(
            cKDTree(to_xy(demand_ll)), SAFE_REACH_M, output_type="coo_matrix").tocsr()
        reach.data[:] = 1.0
        unserved = np.ones(len(demand_ll))
        while len(chosen) < N_SAFE:
            gain = reach @ unserved
            best = int((gain + tie).argmax())
            if gain[best] < 1:
                break
            chosen.append(best)
            unserved[reach[best].indices] = 0.0
    if not chosen:                                      # nothing in reach: highest ground
        chosen = [int(tie.argmax())]

    ids = ids[chosen]
    names = _name_points(lon[ids], lat[ids], places, bbox)
    out = []
    for i, nm in zip(ids, names):
        label = f"High ground, {nm}" + (f" ({elev[i]:.0f} m)" if np.isfinite(elev[i]) else "")
        out.append((label, round(float(lat[i]), 6), round(float(lon[i]), 6)))
    return out


def _write_overlay(path, flood, perm, max_side=1400):
    """RGBA PNG of the model's flood extent for the map (no imaging library needed)."""
    k = max(1, math.ceil(max(flood.shape) / max_side))
    h, w = flood.shape[0] // k * k, flood.shape[1] // k * k
    blk = lambda a: a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)) >= 0.5  # noqa: E731
    fl, pw = blk(flood), blk(perm)
    rgba = np.zeros((*fl.shape, 4), np.uint8)
    rgba[pw] = (36, 82, 160, 120)
    rgba[fl] = (64, 170, 255, 150)
    raw = b"".join(b"\x00" + row.tobytes() for row in rgba)
    chunk = lambda tag, data: (struct.pack(">I", len(data)) + tag + data          # noqa: E731
                               + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", fl.shape[1], fl.shape[0], 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #
def ingest(tif, name, key=None, bbox=None, dem_path=None, threshold=0.5,
           stride_m=STRIDE_M, exposure_m=EXPOSURE_M, safe=None):
    """Build and save a scene from a model GeoTIFF. Returns the scene key."""
    key = key or "sar_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    print(f"Ingesting {tif} as '{key}'")

    flood, perm, tr, bbox = _read_masks(tif, bbox, threshold)
    lat_c = (bbox[1] + bbox[3]) / 2
    px_x = abs(tr.a) * _M_PER_DEG * math.cos(math.radians(lat_c))
    px_y = abs(tr.e) * _M_PER_DEG
    px_m = (px_x + px_y) / 2
    flood_km2 = float(flood.sum()) * px_x * px_y / 1e6
    print(f"  AOI {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}  "
          f"{flood.shape[1]}x{flood.shape[0]} px @ {px_m:.1f} m  flood {flood_km2:.1f} km2")
    if not flood.any():
        raise SystemExit("the model raster has no flood pixels in this area")

    dem = _dem_on_grid(bbox, flood.shape, tr, dem_path)
    depth, depth_source = _depth_cm(flood, perm, dem, px_m)
    marks, wet_ll, stride_used = _marks(flood, perm, depth, dem, tr, px_m, stride_m)
    print(f"  {len(marks)} flood marks at {stride_used:.0f} m spacing  ({depth_source})")

    roads.FETCH_BBOX[key] = (bbox[1], bbox[0], bbox[3], bbox[2])        # S, W, N, E
    g = roads.get_graph(key)
    if g.n_nodes == 0:
        for f in (f"graph_{key}.npz", f"roads_{key}.json"):       # don't cache an empty graph
            os.remove(os.path.join(roads.CACHE_DIR, f))
        roads._GRAPHS.pop(key, None)
        raise SystemExit("OpenStreetMap has no roads in this area - nothing to route on")
    marks["exposed"] = g.nearest_dist(marks[["lon", "lat"]].to_numpy()) <= exposure_m
    print(f"  road graph {g.n_nodes} nodes  -> {int(marks.exposed.sum())} marks within "
          f"{exposure_m:.0f} m of a road")

    places = _fetch_places(bbox)
    demand = marks.loc[marks.exposed if marks.exposed.any() else marks.index,
                       ["lon", "lat"]].to_numpy()
    marks["area"] = _name_points(marks.lon, marks.lat, places, bbox)
    safe_points = ([(n, float(la), float(lo)) for n, la, lo in safe] if safe
                   else _safe_points(g, wet_ll, demand, dem, tr, places, bbox))
    if not safe_points:
        raise SystemExit("no dry road junction found for a safe point - pass --safe")
    for n, la, lo in safe_points:
        print(f"  safe point: {n}  ({la:.4f}, {lo:.4f})")

    os.makedirs(SCENE_DIR, exist_ok=True)
    _write_overlay(os.path.join(SCENE_DIR, f"{key}.png"), flood, perm)
    span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    scene = {
        "meta": {
            "key": key, "name": name,
            "bbox": [round(v, 6) for v in bbox],
            "center": [round(lat_c, 5), round((bbox[0] + bbox[2]) / 2, 5)],
            "zoom": int(min(14, max(8, math.floor(math.log2(360.0 / span)) + 1))),
            "raster": os.path.basename(tif),
            "flood_km2": round(flood_km2, 2),
            "perm_water_km2": round(float(perm.sum()) * px_x * px_y / 1e6, 2),
            "depth_source": depth_source,
            "stride_m": round(stride_used),
            "exposure_m": exposure_m,
            "safe_points": safe_points,
            "safe_source": "given" if safe else "auto: dry high-ground road junctions",
        },
        "marks": {c: [None if isinstance(v, float) and math.isnan(v) else v
                      for v in marks[c].tolist()] for c in marks.columns},
    }
    with gzip.open(os.path.join(SCENE_DIR, f"{key}.json.gz"), "wt") as f:
        json.dump(scene, f)
    print(f"  saved scenes/{key}.json.gz - restart app.py and pick it in the dropdown")
    return key


# --------------------------------------------------------------------------- #
# serving side (no rasterio)                                                  #
# --------------------------------------------------------------------------- #
_LOADED: dict[str, tuple[float, dict, pd.DataFrame]] = {}


def scene_keys() -> list[str]:
    if not os.path.isdir(SCENE_DIR):
        return []
    return sorted(f[:-len(".json.gz")] for f in os.listdir(SCENE_DIR) if f.endswith(".json.gz"))


def _load(key: str):
    path = os.path.join(SCENE_DIR, f"{key}.json.gz")
    mtime = os.path.getmtime(path)
    hit = _LOADED.get(key)
    if hit is None or hit[0] != mtime:
        with gzip.open(path, "rt") as f:
            scene = json.load(f)
        marks = pd.DataFrame(scene["marks"])
        marks["ground_elev_m"] = marks["ground_elev_m"].astype(float)
        marks["locality"] = "-"
        s, w = scene["meta"]["bbox"][1], scene["meta"]["bbox"][0]
        n, e = scene["meta"]["bbox"][3], scene["meta"]["bbox"][2]
        roads.FETCH_BBOX.setdefault(key, (s, w, n, e))
        hit = _LOADED[key] = (mtime, scene["meta"], marks)
    return hit


def scene_marks(key: str) -> pd.DataFrame:
    """Every flood mark of a scene. Columns: lon, lat, depth_cm, ground_elev_m,
    area, locality, exposed."""
    return _load(key)[2]


def scene_config(key: str) -> dict:
    """What routing.compute_routes needs from a city config."""
    m = _load(key)[1]
    return {"key": key, "safe_points": [tuple(p) for p in m["safe_points"]]}


def scene_meta(key: str) -> dict:
    """Front-end description, shaped like cities.city_meta()."""
    m = _load(key)[1]
    w, s, e, n = m["bbox"]
    return {
        "key": key, "name": m["name"], "source": "model",
        "subtitle": f"Flood extent detected by the Sentinel-1 model ({m['raster']})",
        "center": m["center"], "zoom": m["zoom"],
        "safe_points": [{"name": nm, "lat": la, "lon": lo} for nm, la, lo in m["safe_points"]],
        "overlay": {"url": f"/scenes/{key}.png", "bounds": [[s, w], [n, e]]},
        "model": {k: m[k] for k in ("raster", "flood_km2", "perm_water_km2", "depth_source",
                                    "stride_m", "exposure_m", "safe_source")},
    }


# --------------------------------------------------------------------------- #
def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="turn a model GeoTIFF into a routable scene")
    ing.add_argument("tif", help="*_disaster_map.tif or *_flood_prob.tif from the model")
    ing.add_argument("--name", required=True, help='display name, e.g. "Assam, July 2024"')
    ing.add_argument("--key", help="scene id (default: derived from --name)")
    ing.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                     help="area of interest in degrees (default: whole raster, or the "
                          f"most-flooded {AUTO_AOI_KM:.0f} km window of a large one)")
    ing.add_argument("--dem", help="DEM GeoTIFF (default: AWS terrain tiles)")
    ing.add_argument("--threshold", type=float, default=0.5,
                     help="flood probability cut-off, for a *_flood_prob.tif")
    ing.add_argument("--stride-m", type=float, default=STRIDE_M)
    ing.add_argument("--exposure-m", type=float, default=EXPOSURE_M)
    ing.add_argument("--safe", action="append", metavar="NAME,LAT,LON",
                     help="a known shelter / safe point; repeatable (default: auto)")
    sub.add_parser("list", help="show the ingested scenes")
    a = ap.parse_args()

    if a.cmd == "list":
        for k in scene_keys():
            m = _load(k)[1]
            df = scene_marks(k)
            print(f"{k:28} {m['name']:28} {len(df):5d} marks ({int(df.exposed.sum())} exposed)"
                  f"  flood {m['flood_km2']} km2  {len(m['safe_points'])} safe points")
        return
    safe = None
    if a.safe:
        safe = [(n.strip(), float(la), float(lo))
                for n, la, lo in (s.rsplit(",", 2) for s in a.safe)]
    ingest(a.tif, a.name, key=a.key, bbox=tuple(a.bbox) if a.bbox else None,
           dem_path=a.dem, threshold=a.threshold, stride_m=a.stride_m,
           exposure_m=a.exposure_m, safe=safe)


if __name__ == "__main__":
    _main()
