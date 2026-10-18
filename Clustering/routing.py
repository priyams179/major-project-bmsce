"""
routing.py
==========
Evacuation routing on the real road network (see roads.py).

For a given flood we turn the wet marks into a water-depth estimate at every road
node (inverse-distance weighting from the nearest marks). Each road edge then gets
a weight:

    dry            -> its true length in metres
    shallow water  -> length inflated by  (1 + 0.05 * depth_cm)
    deep  water    -> a huge sentinel weight (still traversable so we can tell a
                      zone is "cut off", but never chosen if any dry path exists)

Then ONE multi-source Dijkstra (scipy.sparse.csgraph, C-speed) seeded from every
safe point gives each road node the lowest-risk distance to the nearest safe
point and a predecessor pointer. Each severity zone snaps its centroid to the
nearest road node and follows that chain out - a real turn-by-turn road route.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from roads import get_graph

_EARTH_R = 6_371_000.0
INFLUENCE_M = 360.0        # a mark colours the water field this far out
BLOCK_DEPTH_CM = 250.0     # water over ~2.5 m makes a road truly impassable
WADE_PENALTY = 0.09        # cost multiplier per cm (100 cm -> 10x, 200 cm -> 19x)
_SENTINEL = 5.0e6          # metres-equivalent penalty for an impassable edge
_CUTOFF = 1.0e6            # a route forced across an impassable edge is "cut off"


def _node_depth(coords, mark_lon, mark_lat, mark_depth, mx, my):
    """Inverse-distance water depth (cm) at every road node."""
    tree = cKDTree(np.column_stack([mark_lon * mx, mark_lat * my]))
    node_xy = np.column_stack([coords[:, 1] * mx, coords[:, 0] * my])
    k = min(4, len(mark_depth))
    dist, idx = tree.query(node_xy, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    valid = dist < INFLUENCE_M
    w = np.where(valid, 1.0 / np.maximum(dist, 20.0) ** 2, 0.0)
    dep = mark_depth[np.where(valid, idx, 0)] * w
    wsum = w.sum(axis=1)
    depth = np.where(wsum > 0, dep.sum(axis=1) / np.where(wsum > 0, wsum, 1), 0.0)
    return depth * np.minimum(1.0, valid.sum(axis=1) / 3.0)


def compute_routes(marks_df, zones, city):
    """
    marks_df : DataFrame with lon, lat, depth_cm  (wet marks only)
    zones    : iterable of dicts: zone, tier, fill, centroid_lon, centroid_lat
    city     : the city config dict  (needs key + safe_points)

    Returns (features, info_by_zone).
    """
    safe_pts = city.get("safe_points", [])
    mlon = marks_df["lon"].to_numpy(float)
    mlat = marks_df["lat"].to_numpy(float)
    mdep = marks_df["depth_cm"].to_numpy(float)
    if len(mlon) == 0 or not safe_pts:
        return [], {}

    g = get_graph(city["key"])
    coords = g.coords                                   # (N,2) lat,lon
    lat0 = float(coords[:, 0].mean())
    mx = np.radians(1.0) * np.cos(np.radians(lat0)) * _EARTH_R
    my = np.radians(1.0) * _EARTH_R

    node_depth = _node_depth(coords, mlon, mlat, mdep, mx, my)

    src = np.repeat(np.arange(g.n_nodes), np.diff(g.indptr))
    edge_depth = np.maximum(node_depth[src], node_depth[g.indices])
    blocked = edge_depth > BLOCK_DEPTH_CM
    w = np.where(blocked,
                 g.seg_len + _SENTINEL,
                 g.seg_len * (1.0 + WADE_PENALTY * edge_depth))
    graph = csr_matrix((w, g.indices, g.indptr), shape=(g.n_nodes, g.n_nodes))

    safe_nodes = g.snap_many(np.array([[lon, lat] for (_, lat, lon) in safe_pts]))
    dist, pred, sources = dijkstra(graph, directed=False, indices=safe_nodes,
                                   min_only=True, return_predecessors=True)
    node_to_safe = {int(n): i for i, n in enumerate(safe_nodes)}

    feats, info = [], {}
    for z in zones:
        zid = int(z["zone"])
        start = g.snap(float(z["centroid_lon"]), float(z["centroid_lat"]))
        if not np.isfinite(dist[start]) or dist[start] >= _CUTOFF:
            info[zid] = {"reachable": False, "distance_km": None,
                         "max_depth_cm": None, "safe_name": None, "bearing": None}
            continue

        chain = [start]
        cur = start
        while pred[cur] >= 0:
            cur = int(pred[cur])
            chain.append(cur)
        pll = coords[chain]                             # (M,2) lat,lon
        route_coords = [[float(lon), float(lat)] for lat, lon in pll]

        d = np.radians(np.diff(pll, axis=0))
        la = np.radians(pll[:, 0])
        hav = (np.sin(d[:, 0] / 2) ** 2
               + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(d[:, 1] / 2) ** 2)
        km = float(_EARTH_R * 2 * np.arcsin(np.sqrt(hav)).sum() / 1000.0)

        si = node_to_safe.get(int(sources[start]), -1)
        safe_name = safe_pts[si][0] if si >= 0 else "safe ground"
        h = pll[min(3, len(pll) - 1)]
        brg = _bearing(h[0], h[1], pll[-1, 0], pll[-1, 1])

        info[zid] = {
            "reachable": True,
            "distance_km": round(km, 2),
            "max_depth_cm": round(float(node_depth[chain].max()), 1),
            "safe_name": safe_name,
            "bearing": brg,
        }
        feats.append({
            "type": "Feature",
            "properties": {"zone": zid, "tier": z["tier"], "color": z["fill"],
                           "distance_km": info[zid]["distance_km"],
                           "max_depth_cm": info[zid]["max_depth_cm"],
                           "safe_name": safe_name, "bearing": brg},
            "geometry": {"type": "LineString", "coordinates": route_coords},
        })
    return feats, info


def _bearing(lat1, lon1, lat2, lon2):
    d = np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((d + 22.5) // 45) % 8]
