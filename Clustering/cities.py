"""
cities.py
=========
Synthetic-but-realistic flood generators for several real cities.

For each city we seed candidate "flood marks" across the neighbourhoods that
actually flood (from news coverage / flood studies of past events), each mark
carrying a synthetic **ground elevation**. A flood is simulated by raising one
number - the flood *stage* in metres at the reference gauge:

    depth at a mark  =  stage - ground_elevation      (kept only where > 0)

Low-lying marks go under first; as the stage rises the water spreads and more
marks light up. Those inundated marks are what `flood_zones.py` clusters, scores
and tiers.

Coordinates and neighbourhood names are real. Elevations, mark counts and the
stage->depth mapping are synthetic and tuned for a legible demo - this is NOT a
hydrological model.

Area tuple: (name, locality, lat, lon, n_pts, spread_deg,
             base_elev_m, elev_gain_m_per_km, elev_noise_m)
  base_elev = ground height at the pocket's low point (waterfront / marsh)
  elev_gain = how fast the ground rises as you move away from that point
"""

from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_R = 6_371_000.0


# =========================================================================== #
# NEW YORK CITY - Atlantic hurricane storm surge (Sandy / FEMA flood zones)     #
# =========================================================================== #
NYC = {
    "key": "nyc",
    "name": "New York City",
    "subtitle": "Atlantic hurricane storm surge - tidal coast & rivers",
    "center": [40.700, -73.930], "zoom": 11,
    "driver": "storm surge",
    "stage_label": "Water level above tidal datum",
    "stage_min": 0.4, "stage_max": 4.5, "stage_default": 2.5,
    "locality_label": "borough",
    "noise_bbox": [-74.02, 40.55, -73.78, 40.82],
    "references": [
        {"stage_m": 0.8, "label": "Nuisance / king tide"},
        {"stage_m": 1.5, "label": "Moderate coastal storm"},
        {"stage_m": 2.1, "label": "Hurricane Irene (2011)"},
        {"stage_m": 3.4, "label": "Hurricane Sandy (2012), the Battery"},
        {"stage_m": 4.0, "label": "Design storm / worst case"},
    ],
    "areas": [
        ("Battery Park & Financial District", "Manhattan", 40.7033, -74.0140, 120, 0.0060, 0.5, 3.2, 0.25),
        ("Lower East Side / East River Park", "Manhattan", 40.7145, -73.9760, 110, 0.0075, 0.9, 3.0, 0.25),
        ("Two Bridges / South St Seaport",    "Manhattan", 40.7092, -73.9995, 70,  0.0040, 0.8, 3.4, 0.20),
        ("Red Hook",                          "Brooklyn",  40.6740, -74.0110, 95,  0.0055, 0.4, 2.6, 0.20),
        ("Gowanus Canal",                     "Brooklyn",  40.6730, -73.9950, 80,  0.0050, 0.7, 2.8, 0.20),
        ("DUMBO / Brooklyn Navy Yard",        "Brooklyn",  40.7015, -73.9770, 85,  0.0070, 0.8, 3.0, 0.22),
        ("Greenpoint / Newtown Creek",        "Brooklyn",  40.7310, -73.9560, 85,  0.0075, 1.0, 2.8, 0.25),
        ("Coney Island & Sea Gate",           "Brooklyn",  40.5760, -73.9820, 120, 0.0110, 0.6, 2.2, 0.20),
        ("Sheepshead Bay / Gerritsen",        "Brooklyn",  40.5860, -73.9420, 90,  0.0090, 0.5, 2.0, 0.20),
        ("Hunts Point",                       "Bronx",     40.8080, -73.8790, 70,  0.0065, 1.1, 3.0, 0.25),
        ("East Elmhurst / LaGuardia",         "Queens",    40.7690, -73.8720, 80,  0.0080, 0.6, 2.6, 0.22),
        ("Hamilton Beach / Howard Beach",     "Queens",    40.6570, -73.8380, 85,  0.0075, 0.35, 1.8, 0.18),
        ("Broad Channel",                     "Queens",    40.6080, -73.8200, 60,  0.0060, 0.2, 1.4, 0.15),
        ("Rockaway Peninsula",                "Queens",    40.5880, -73.8150, 140, 0.0180, 0.5, 1.6, 0.20),
        ("Midland & South Beach",             "Staten Is", 40.5850, -74.0770, 110, 0.0110, 0.4, 1.9, 0.20),
        ("Oakwood Beach",                     "Staten Is", 40.5560, -74.1150, 70,  0.0080, 0.3, 1.7, 0.18),
    ],
    # high ground people actually evacuate to (parks on the terminal moraine / hills)
    "safe_points": [
        ("Central Park, Manhattan",      40.7850, -73.9660),
        ("Prospect Park, Brooklyn",      40.6600, -73.9700),
        ("Forest Park, Queens",          40.7010, -73.8600),
        ("Todt Hill, Staten Island",     40.6010, -74.1040),
        ("Van Cortlandt Park, Bronx",    40.8970, -73.8860),
    ],
}


# =========================================================================== #
# CHENNAI - NE-monsoon deluge + Chembarambakkam release into the Adyar (2015)   #
# =========================================================================== #
CHENNAI = {
    "key": "chennai",
    "name": "Chennai",
    "subtitle": "NE-monsoon deluge & reservoir release - Adyar / Cooum / Pallikaranai",
    "center": [13.020, 80.210], "zoom": 11,
    "driver": "flood stage",
    "stage_label": "Peak flood-water level",
    "stage_min": 0.3, "stage_max": 3.6, "stage_default": 2.0,
    "locality_label": "basin",
    "noise_bbox": [80.06, 12.85, 80.31, 13.24],
    "references": [
        {"stage_m": 0.6, "label": "Heavy NE-monsoon spell"},
        {"stage_m": 1.2, "label": "2021 Nov flooding"},
        {"stage_m": 2.0, "label": "Cyclone Michaung (Dec 2023)"},
        {"stage_m": 2.8, "label": "Dec 2015 floods (Velachery ~1.2 m+)"},
        {"stage_m": 3.4, "label": "Chembarambakkam peak release"},
    ],
    "areas": [
        ("Velachery",                     "Kovalam basin", 12.9755, 80.2210, 130, 0.0090, 0.25, 1.9, 0.20),
        ("Pallikaranai Marsh",            "Kovalam basin", 12.9360, 80.2120, 120, 0.0130, 0.10, 1.4, 0.15),
        ("Perungudi / Thoraipakkam",      "Kovalam basin", 12.9600, 80.2400, 90,  0.0090, 0.35, 1.8, 0.20),
        ("Madipakkam",                    "Kovalam basin", 12.9620, 80.1980, 85,  0.0080, 0.4,  2.0, 0.20),
        ("Mudichur",                      "Adyar",         12.9130, 80.0660, 95,  0.0090, 0.3,  1.7, 0.20),
        ("West Tambaram / Varadharajapuram","Adyar",       12.9180, 80.1010, 90,  0.0100, 0.4,  1.8, 0.20),
        ("Saidapet & Jones Road",         "Adyar",         13.0215, 80.2225, 85,  0.0055, 0.5,  2.4, 0.22),
        ("Kotturpuram & MRC Nagar",       "Adyar",         13.0150, 80.2470, 80,  0.0060, 0.45, 2.6, 0.20),
        ("Adyar / Ferry Road",            "Adyar",         13.0030, 80.2560, 75,  0.0060, 0.5,  2.6, 0.20),
        ("T. Nagar & Mambalam",           "Adyar",         13.0410, 80.2340, 80,  0.0060, 0.8,  2.8, 0.22),
        ("Mylapore & Raja Annamalaipuram","Adyar",         13.0330, 80.2690, 60,  0.0050, 0.9,  3.0, 0.20),
        ("Cooum banks - Chetpet/Egmore",  "Cooum",         13.0760, 80.2450, 90,  0.0090, 0.7,  2.6, 0.22),
        ("Aminjikarai & Koyambedu",       "Cooum",         13.0730, 80.2200, 75,  0.0080, 0.9,  2.7, 0.22),
        ("Ambattur & Mogappair",          "Cooum",         13.1010, 80.1620, 85,  0.0110, 1.0,  2.8, 0.25),
        ("Manali & Madhavaram",           "Kosasthalaiyar",13.1650, 80.2600, 80,  0.0110, 0.6,  2.4, 0.22),
        ("Ennore / Wimco Nagar",          "Kosasthalaiyar",13.2150, 80.3210, 70,  0.0100, 0.4,  1.9, 0.20),
        ("Semmancheri & Sholinganallur",  "Kovalam basin", 12.8880, 80.2270, 80,  0.0110, 0.35, 1.8, 0.20),
        ("Guindy & Ekkatuthangal",        "Adyar",         13.0090, 80.2110, 70,  0.0065, 0.7,  2.6, 0.22),
    ],
    "safe_points": [
        ("St. Thomas Mount",             13.0075, 80.1960),
        ("Pallavaram tableland",         12.9670, 80.1500),
        ("Anna Nagar / Nolambur",        13.0870, 80.1900),
        ("Red Hills ridge",              13.1900, 80.1800),
        ("Tambaram Sanatorium ground",   12.9250, 80.1150),
    ],
}


# =========================================================================== #
# CUTTACK - Mahanadi delta; city pinned between Mahanadi & Kathajodi rivers     #
# =========================================================================== #
CUTTACK = {
    "key": "cuttack",
    "name": "Cuttack",
    "subtitle": "Mahanadi-delta river flood - city wedged between Mahanadi & Kathajodi",
    "center": [20.472, 85.879], "zoom": 12,
    "driver": "river flood stage",
    "stage_label": "River level above danger mark",
    "stage_min": 0.3, "stage_max": 4.0, "stage_default": 2.2,
    "locality_label": "river",
    "noise_bbox": [85.79, 20.42, 85.95, 20.53],
    "references": [
        {"stage_m": 0.5, "label": "Above danger mark at Naraj"},
        {"stage_m": 1.2, "label": "2011 Mahanadi flood"},
        {"stage_m": 2.0, "label": "2022 Odisha floods"},
        {"stage_m": 2.8, "label": "2001-style high flood"},
        {"stage_m": 3.6, "label": "Hirakud max discharge scenario"},
    ],
    "areas": [
        ("Naraj (river bifurcation)",   "Mahanadi",  20.5050, 85.8150, 80,  0.0075, 0.25, 2.4, 0.20),
        ("Jobra & Barrage",             "Mahanadi",  20.4790, 85.8950, 75,  0.0055, 0.35, 2.8, 0.20),
        ("Sikharpur & Ranihat",         "Mahanadi",  20.4880, 85.8680, 85,  0.0060, 0.5,  3.0, 0.22),
        ("Choudhury Bazar & Chandni Chowk","Mahanadi",20.4780, 85.8850, 70, 0.0045, 0.7,  3.2, 0.20),
        ("Nuabazar & Dargha Bazar",     "Mahanadi",  20.4720, 85.8880, 70,  0.0045, 0.65, 3.2, 0.20),
        ("Badambadi & Link Road",       "Mahanadi",  20.4760, 85.8590, 80,  0.0060, 0.8,  3.0, 0.22),
        ("Matagajpur",                  "Kathajodi", 20.4460, 85.8580, 75,  0.0060, 0.2,  2.2, 0.18),
        ("Tarapur & OMP Square",        "Kathajodi", 20.4560, 85.8720, 80,  0.0060, 0.35, 2.4, 0.20),
        ("Kafla & Nuapatna",            "Kathajodi", 20.4600, 85.8850, 70,  0.0055, 0.4,  2.5, 0.20),
        ("Bidanasi / CDA reclaimed",    "Kathajodi", 20.4520, 85.9010, 90,  0.0080, 0.3,  2.0, 0.20),
        ("Gandarpur & Gopalpur",        "Kathajodi", 20.4680, 85.9060, 75,  0.0065, 0.45, 2.3, 0.20),
        ("Trisulia & Phulnakhara",      "Kathajodi", 20.4980, 85.9080, 70,  0.0080, 0.4,  2.2, 0.20),
        ("Jagatpur (north bank)",       "Mahanadi",  20.5200, 85.8700, 80,  0.0090, 0.3,  2.2, 0.20),
        ("Raghunathpur riverside",      "Kathajodi", 20.4390, 85.8900, 60,  0.0060, 0.15, 1.9, 0.15),
    ],
    # the city is an island between two rivers; safe ground is across a bridge
    "safe_points": [
        ("Higher ground toward Bhubaneswar (S)", 20.3700, 85.8450),
        ("Barang uplands (SW)",                  20.4150, 85.7850),
        ("Choudwar ridge (N)",                   20.5450, 85.8950),
        ("Salipur side (E, past confluence)",    20.4760, 85.9900),
    ],
}


CITIES = {c["key"]: c for c in (NYC, CHENNAI, CUTTACK)}
DEFAULT_CITY = "nyc"


# --------------------------------------------------------------------------- #
def _scatter(n, spread, rng):
    """Elliptical Gaussian scatter, slightly elongated along the shore/bank."""
    return rng.normal(0, spread, n), rng.normal(0, spread * 0.62, n)


def generate_candidates(city_key: str, seed: int = 42) -> pd.DataFrame:
    """Every *potential* flood mark for a city, with a synthetic ground elevation."""
    city = CITIES[city_key]
    rng = np.random.default_rng(seed)
    frames = []
    for name, loc, lat0, lon0, n, spread, base, gain, noise in city["areas"]:
        dlon, dlat = _scatter(n, spread, rng)
        lon, lat = lon0 + dlon, lat0 + dlat
        x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_R
        y = np.radians(lat - lat0) * EARTH_R
        dist_km = np.sqrt(x**2 + y**2) / 1000.0
        elev = np.clip(base + gain * dist_km + rng.normal(0, noise, n), 0.03, None)
        frames.append(pd.DataFrame({
            "lon": lon, "lat": lat,
            "ground_elev_m": elev.round(2),
            "area": name, "locality": loc,
        }))
    x0, y0, x1, y1 = city["noise_bbox"]
    n_noise = 40
    frames.append(pd.DataFrame({
        "lon": rng.uniform(x0, x1, n_noise),
        "lat": rng.uniform(y0, y1, n_noise),
        "ground_elev_m": rng.uniform(0.8, 3.5, n_noise).round(2),
        "area": "scattered", "locality": "-",
    }))
    return pd.concat(frames, ignore_index=True)


def inundated_marks(city_key: str, stage_m: float, seed: int = 42) -> pd.DataFrame:
    """Flood marks under water at the given stage. Columns: lon, lat, depth_cm,
    ground_elev_m, area, locality."""
    cand = generate_candidates(city_key, seed)
    depth_m = stage_m - cand["ground_elev_m"].to_numpy()
    wet = depth_m > 0.03
    out = cand.loc[wet].copy()
    out["depth_cm"] = (depth_m[wet] * 100.0).round(1)
    return out.reset_index(drop=True)


def city_meta(city_key: str) -> dict:
    """The front-end-facing description of a city (no geometry)."""
    c = CITIES[city_key]
    m = {k: c[k] for k in (
        "key", "name", "subtitle", "center", "zoom", "driver",
        "stage_label", "stage_min", "stage_max", "stage_default",
        "locality_label", "references",
    )}
    m["safe_points"] = [{"name": n, "lat": lat, "lon": lon}
                        for (n, lat, lon) in c.get("safe_points", [])]
    return m


if __name__ == "__main__":
    for key in CITIES:
        cand = generate_candidates(key)
        print(f"\n{CITIES[key]['name']}: {len(cand)} candidate marks, "
              f"{len(CITIES[key]['areas'])} pockets")
        for s in (0.8, 1.5, 2.5, 3.5):
            m = inundated_marks(key, s)
            md = f"{m.depth_cm.max():.0f}" if len(m) else "-"
            print(f"   stage {s:>4} m -> {len(m):4d} wet, max depth {md} cm")
