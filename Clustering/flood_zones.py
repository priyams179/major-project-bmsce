"""
flood_zones.py
==============
Identify flood *severity zones* from point "flood marks" on a map.

Pipeline
--------
1. SPATIAL CLUSTERING  -> group nearby marks into contiguous flooded footprints
   (DBSCAN with the haversine metric, so `eps` is a real ground distance in metres).
2. SEVERITY SCORING    -> for each zone, combine how *deep* the water is (mark
   attribute, e.g. water depth) with how *concentrated* the marks are (density).
3. TIERING             -> map the composite score to Critical / High / Moderate / Low.
4. OUTPUT              -> labelled points, a zone summary table, and GeoJSON polygons
   you can drop straight onto a map (Leaflet, QGIS, geojson.io, etc.).

Why DBSCAN? Flood marks are irregular blobs, the number of zones is unknown, and
some marks are stray/noise. DBSCAN finds arbitrarily-shaped dense regions, needs
no k, and labels sparse stray marks as noise (-1) instead of forcing them in.

Dependencies: numpy, pandas, scikit-learn, scipy  (no GIS libs required).
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Geometry helpers (lat/lon -> local metres, polygon area, convex hull)       #
# --------------------------------------------------------------------------- #
def _local_xy_metres(lon, lat, lon0, lat0):
    """Equirectangular projection to metres around a reference point.
    Fine for city / district scale zones."""
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def _polygon_area_km2(lon, lat):
    """Convex-hull area of a set of points, in km^2."""
    if len(lon) < 3:
        return 0.0
    x, y = _local_xy_metres(lon, lat, lon.mean(), lat.mean())
    pts = np.column_stack([x, y])
    try:
        return ConvexHull(pts).volume / 1e6  # 2D 'volume' == area
    except Exception:
        return 0.0


def _hull_ring_lonlat(lon, lat):
    """Convex-hull boundary as a closed [lon,lat] ring for GeoJSON."""
    lon = np.asarray(lon); lat = np.asarray(lat)
    if len(lon) < 3:
        # degenerate: return the points themselves (a dot or line)
        ring = list(zip(lon.tolist(), lat.tolist()))
        return ring + ring[:1]
    pts = np.column_stack([lon, lat])
    hull = ConvexHull(pts)
    ring = pts[hull.vertices]
    ring = np.vstack([ring, ring[0]])            # close the ring
    return ring.tolist()


# --------------------------------------------------------------------------- #
# Main estimator                                                              #
# --------------------------------------------------------------------------- #
class FloodSeverityZones:
    """
    Parameters
    ----------
    eps_m : float
        DBSCAN neighbourhood radius in METRES. Two marks closer than this are
        considered part of the same flooded area. Start ~ the spacing between
        your marks (e.g. 150-300 m for street-level survey points).
    min_samples : int
        Minimum marks to seed a zone. Higher -> only well-attested zones survive,
        more marks flagged as noise.
    severity_col : str | None
        Column holding the per-mark severity (e.g. water depth in cm). If None,
        zones are ranked by mark density alone.
    severity_influence : float
        0.0  -> cluster on location only (severity used just to rank zones).
        >0   -> severity is added as an extra clustering dimension, so a deep
                pocket splits off from adjacent shallow water even when they touch.
                Try 0.5-1.0 if you want depth to shape the zone boundaries.
    depth_weight : float
        Weight of (normalised) depth vs (normalised) density in the zone score.
        0.7 -> depth-led, 0.3 -> density-led.
    """

    TIERS = ["Low", "Moderate", "High", "Critical"]

    def __init__(self, eps_m=200.0, min_samples=5, severity_col="depth_cm",
                 severity_influence=0.0, depth_weight=0.7):
        self.eps_m = eps_m
        self.min_samples = min_samples
        self.severity_col = severity_col
        self.severity_influence = severity_influence
        self.depth_weight = depth_weight

    # -- clustering --------------------------------------------------------- #
    def _cluster(self, df):
        coords_rad = np.radians(df[["lat", "lon"]].to_numpy())
        eps_rad = self.eps_m / EARTH_RADIUS_M

        if self.severity_influence > 0 and self.severity_col in df:
            # Augment with a scaled severity axis. We convert the severity's
            # z-score into a "pseudo-distance" and merge it with the haversine
            # ground distance via a precomputed distance matrix.
            from sklearn.metrics import pairwise_distances
            geo = pairwise_distances(coords_rad, metric="haversine")  # radians
            s = df[self.severity_col].to_numpy(float)
            s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)
            sev = np.abs(s[:, None] - s[None, :]) * eps_rad * self.severity_influence
            dmat = np.sqrt(geo**2 + sev**2)
            labels = DBSCAN(eps=eps_rad, min_samples=self.min_samples,
                            metric="precomputed").fit_predict(dmat)
        else:
            labels = DBSCAN(eps=eps_rad, min_samples=self.min_samples,
                            metric="haversine").fit_predict(coords_rad)
        return labels

    # -- fit ---------------------------------------------------------------- #
    def fit(self, df: pd.DataFrame):
        """`df` needs columns: lon, lat, and (optionally) the severity column."""
        df = df.copy()
        df["zone"] = self._cluster(df)
        self.points_ = df

        rows = []
        for z, g in df[df.zone >= 0].groupby("zone"):
            area = _polygon_area_km2(g.lon.to_numpy(), g.lat.to_numpy())
            rec = {
                "zone": int(z),
                "n_marks": len(g),
                "area_km2": round(area, 4),
                "density_marks_per_km2": round(len(g) / area, 1) if area > 0 else np.nan,
                "centroid_lon": round(g.lon.mean(), 6),
                "centroid_lat": round(g.lat.mean(), 6),
            }
            if self.severity_col in g:
                rec["mean_depth"] = round(g[self.severity_col].mean(), 1)
                rec["max_depth"] = round(g[self.severity_col].max(), 1)
                rec["p90_depth"] = round(g[self.severity_col].quantile(0.9), 1)
            rows.append(rec)

        zones = pd.DataFrame(rows)
        self.zones_ = self._score_and_tier(zones)
        self.n_noise_ = int((df.zone == -1).sum())
        return self

    # -- scoring & tiering -------------------------------------------------- #
    @staticmethod
    def _norm(x):
        x = np.asarray(x, float)
        rng = np.nanmax(x) - np.nanmin(x)
        return np.zeros_like(x) if rng == 0 else (x - np.nanmin(x)) / rng

    def _score_and_tier(self, zones):
        if zones.empty:
            return zones
        density_n = self._norm(zones["density_marks_per_km2"].fillna(0))
        if "mean_depth" in zones:
            # depth component blends typical depth (mean) and worst-case (p90)
            depth_n = self._norm(0.5 * zones["mean_depth"] + 0.5 * zones["p90_depth"])
        else:
            depth_n = np.zeros(len(zones))
            self.depth_weight = 0.0

        zones["severity_score"] = (self.depth_weight * depth_n
                                   + (1 - self.depth_weight) * density_n).round(3)

        # Rank into tiers by score quantiles (robust to outliers). With few
        # zones, quantile cuts still give a sensible ordering.
        q = zones["severity_score"].rank(pct=True)
        tier = np.select(
            [q >= 0.75, q >= 0.50, q >= 0.25],
            ["Critical", "High", "Moderate"], default="Low")
        zones["tier"] = tier
        return zones.sort_values("severity_score", ascending=False).reset_index(drop=True)

    # -- export ------------------------------------------------------------- #
    def to_geojson(self, path=None):
        """Convex-hull polygon per zone, coloured by tier."""
        colour = {"Critical": "#7f0000", "High": "#d7301f",
                  "Moderate": "#fc8d59", "Low": "#fdcc8a"}
        feats = []
        for _, zr in self.zones_.iterrows():
            g = self.points_[self.points_.zone == zr["zone"]]
            ring = _hull_ring_lonlat(g.lon.to_numpy(), g.lat.to_numpy())
            props = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                     for k, v in zr.items()}
            props["fill"] = colour.get(zr["tier"], "#fdcc8a")
            feats.append({"type": "Feature",
                          "properties": props,
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
        fc = {"type": "FeatureCollection", "features": feats}
        if path:
            with open(path, "w") as f:
                json.dump(fc, f, indent=2)
        return fc


# --------------------------------------------------------------------------- #
# Demo on synthetic flood marks so the file runs standalone                    #
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(7)
    # Three flooded pockets around a city centre, plus scattered stray marks.
    def blob(lon0, lat0, n, spread, depth_mu, depth_sd):
        return pd.DataFrame({
            "lon": rng.normal(lon0, spread, n),
            "lat": rng.normal(lat0, spread, n),
            "depth_cm": np.clip(rng.normal(depth_mu, depth_sd, n), 0, None),
        })

    marks = pd.concat([
        blob(80.240, 13.060, 60, 0.0015, 95, 25),  # deep, dense -> Critical
        blob(80.270, 13.085, 45, 0.0018, 55, 18),  # moderate
        blob(80.210, 13.040, 30, 0.0020, 30, 12),  # shallow
        pd.DataFrame({"lon": rng.uniform(80.19, 80.29, 15),   # stray noise
                      "lat": rng.uniform(13.02, 13.10, 15),
                      "depth_cm": rng.uniform(2, 20, 15)}),
    ], ignore_index=True)

    model = FloodSeverityZones(eps_m=250, min_samples=6,
                               severity_col="depth_cm", depth_weight=0.7)
    model.fit(marks)

    print(f"Total marks: {len(marks)} | zones found: {len(model.zones_)} | "
          f"noise marks: {model.n_noise_}\n")
    cols = ["zone", "tier", "severity_score", "n_marks", "area_km2",
            "density_marks_per_km2", "mean_depth", "max_depth"]
    print(model.zones_[cols].to_string(index=False))
    model.to_geojson("flood_zones.geojson")
    print("\nGeoJSON written -> flood_zones.geojson")


if __name__ == "__main__":
    _demo()
