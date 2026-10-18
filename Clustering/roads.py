"""
roads.py
========
Real road network per city, for evacuation routing.

The raw OpenStreetMap road geometry lives in `cache/roads_<city>.json` (an
Overpass `out skel` dump). The first time a city is used we parse that into a
compact, undirected graph and cache it as `cache/graph_<city>.npz`:

    coords   (N, 2) float32   node lat, lon
    indptr   (N+1,) int64     CSR row pointers
    indices  (E2,)  int32     CSR neighbour node ids   (each edge stored both ways)
    seg_len  (E2,)  float32   edge length in metres

`RoadGraph` also carries a KD-tree (in local metres) so we can snap an arbitrary
lat/lon - a zone centroid, a safe point - onto the nearest road node.

If the raw dump is missing, `fetch_roads()` pulls it from Overpass (needs
internet, one-off).
"""

from __future__ import annotations
import gzip
import json
import os
import urllib.parse
import urllib.request

import numpy as np
from scipy.spatial import cKDTree

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
_EARTH_R = 6_371_000.0

# bounding boxes used for the Overpass fetch  (S, W, N, E)
FETCH_BBOX = {
    "nyc":     (40.52, -74.27, 40.92, -73.71),
    "chennai": (12.80, 80.02, 13.28, 80.36),
    "cuttack": (20.40, 85.76, 20.56, 85.99),
}
_HW = ("motorway|trunk|primary|secondary|tertiary|residential|unclassified|"
       "living_street|motorway_link|trunk_link|primary_link|secondary_link|"
       "tertiary_link")
_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# overpass-api.de answers 406 to urllib's default User-Agent
_UA = {"User-Agent": "flood-zones-routing/1.0"}

_GRAPHS: dict[str, "RoadGraph"] = {}


# --------------------------------------------------------------------------- #
def fetch_roads(city_key: str) -> str:
    """Download the Overpass road dump for a city into cache/. Returns the path."""
    s, w, n, e = FETCH_BBOX[city_key]
    q = (f"[out:json][timeout:170];"
         f'way["highway"~"^({_HW})$"]({s},{w},{n},{e});(._;>;);out skel qt;')
    body = urllib.parse.urlencode({"data": q}).encode()
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"roads_{city_key}.json")
    last = None
    for host in _MIRRORS:
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(host, data=body, headers=_UA), timeout=180) as r:
                raw = r.read()
            if raw[:1] not in (b"{", b"["):
                last = RuntimeError(f"{host}: non-JSON reply")
                continue
            with open(path, "wb") as f:
                f.write(raw)
            return path
        except Exception as exc:            # noqa: BLE001
            last = exc
    raise RuntimeError(f"could not fetch roads for {city_key}: {last}")


def _raw_path(city_key: str) -> str:
    for name in (f"roads_{city_key}.json", f"roads_{city_key}.json.gz"):
        p = os.path.join(CACHE_DIR, name)
        if os.path.exists(p):
            return p
    return fetch_roads(city_key)


def _load_overpass(city_key: str) -> dict:
    p = _raw_path(city_key)
    opn = gzip.open if p.endswith(".gz") else open
    with opn(p, "rt") as f:
        return json.load(f)


def _build_npz(city_key: str) -> dict:
    data = _load_overpass(city_key)
    node_lat: dict[int, float] = {}
    node_lon: dict[int, float] = {}
    ways: list[list[int]] = []
    for el in data["elements"]:
        t = el["type"]
        if t == "node":
            node_lat[el["id"]] = el["lat"]
            node_lon[el["id"]] = el["lon"]
        elif t == "way" and len(el.get("nodes", ())) > 1:
            ways.append(el["nodes"])

    remap: dict[int, int] = {}
    lat_list: list[float] = []
    lon_list: list[float] = []

    def local_id(osm_id: int) -> int:
        i = remap.get(osm_id)
        if i is None:
            i = len(lat_list)
            remap[osm_id] = i
            lat_list.append(node_lat[osm_id])
            lon_list.append(node_lon[osm_id])
        return i

    pairs: set[tuple[int, int]] = set()
    for w in ways:
        prev = None
        for osm_id in w:
            if osm_id not in node_lat:
                prev = None
                continue
            cur = local_id(osm_id)
            if prev is not None and prev != cur:
                pairs.add((prev, cur) if prev < cur else (cur, prev))
            prev = cur

    coords = np.column_stack([np.asarray(lat_list, np.float64),
                              np.asarray(lon_list, np.float64)])
    e = np.fromiter((x for p in pairs for x in p), np.int64,
                    count=2 * len(pairs)).reshape(-1, 2)

    la = np.radians(coords[:, 0]); lo = np.radians(coords[:, 1])
    dlat = la[e[:, 1]] - la[e[:, 0]]
    dlon = lo[e[:, 1]] - lo[e[:, 0]]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(la[e[:, 0]]) * np.cos(la[e[:, 1]]) * np.sin(dlon / 2) ** 2)
    seg = _EARTH_R * 2 * np.arcsin(np.sqrt(a))

    # undirected CSR: add both directions, sort by source
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    wgt = np.concatenate([seg, seg])
    order = np.argsort(src, kind="stable")
    src, dst, wgt = src[order], dst[order], wgt[order]
    n = len(coords)
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)

    out = {
        "coords": coords.astype(np.float32),
        "indptr": indptr,
        "indices": dst.astype(np.int32),
        "seg_len": wgt.astype(np.float32),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(os.path.join(CACHE_DIR, f"graph_{city_key}.npz"), **out)
    return out


# --------------------------------------------------------------------------- #
class RoadGraph:
    def __init__(self, city_key: str):
        npz = os.path.join(CACHE_DIR, f"graph_{city_key}.npz")
        d = dict(np.load(npz)) if os.path.exists(npz) else _build_npz(city_key)
        self.coords = d["coords"]                     # (N,2) lat,lon
        self.indptr = d["indptr"]
        self.indices = d["indices"]
        self.seg_len = d["seg_len"].astype(np.float64)
        lat0 = float(self.coords[:, 0].mean())
        self._mx = np.radians(1.0) * np.cos(np.radians(lat0)) * _EARTH_R
        self._my = np.radians(1.0) * _EARTH_R
        xy = np.column_stack([self.coords[:, 1] * self._mx,
                              self.coords[:, 0] * self._my])
        self._tree = cKDTree(xy)

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    def snap(self, lon: float, lat: float) -> int:
        return int(self._tree.query([lon * self._mx, lat * self._my])[1])

    def snap_many(self, lonlat: np.ndarray) -> np.ndarray:
        q = np.column_stack([lonlat[:, 0] * self._mx, lonlat[:, 1] * self._my])
        return self._tree.query(q)[1].astype(int)

    def nearest_dist(self, lonlat: np.ndarray) -> np.ndarray:
        """Distance in metres from each lon/lat to its nearest road node."""
        q = np.column_stack([lonlat[:, 0] * self._mx, lonlat[:, 1] * self._my])
        return self._tree.query(q)[0]

    def snap_knn(self, lon: float, lat: float, k: int) -> np.ndarray:
        """The k nearest road nodes to a point, closest first."""
        k = min(k, self.n_nodes)
        idx = self._tree.query([lon * self._mx, lat * self._my], k=k)[1]
        return np.atleast_1d(idx).astype(int)


def get_graph(city_key: str) -> RoadGraph:
    g = _GRAPHS.get(city_key)
    if g is None:
        g = _GRAPHS[city_key] = RoadGraph(city_key)
    return g


if __name__ == "__main__":
    for key in FETCH_BBOX:
        g = get_graph(key)
        print(f"{key:8}  {g.n_nodes:>7} nodes  {len(g.indices)//2:>7} edges")
