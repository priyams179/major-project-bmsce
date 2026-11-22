"""Static configuration: events, imagery windows, bands, paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "landsentry"
CKPT_DIR = ROOT / "checkpoints" / "landsentry"
RESULTS_DIR = ROOT / "results" / "landsentry"
PRED_DIR = DATA_DIR / "preds"          # full-scene probability maps (large, git-ignored)

# Earth Engine cloud project (same one the rest of the repo uses).
GEE_PROJECT = os.environ.get("LANDSENTRY_GEE_PROJECT", "major-project-512310")

# Sentinel-2 Level-1C (TOA) is used instead of L2A because S2_SR coverage over
# India is incomplete before Dec 2018 (14 L2A vs 54 L1C scenes over Kodagu for
# Jan-May 2018), and both events must be processed identically.
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"
S2_CLOUD_PROB = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_PROB_MAX = 40          # s2cloudless probability threshold (%)
SCENE_CLOUD_MAX = 70         # CLOUDY_PIXEL_PERCENTAGE pre-filter

# 10 m bands, model channel order: blue, green, red, NIR (per date).
BANDS = ("B2", "B3", "B4", "B8")
RGB_IDX = (2, 1, 0)          # indices of R, G, B inside BANDS
NIR_IDX, RED_IDX = 3, 2
SCALE_M = 10
UTM_EPSG = 32643             # UTM 43N covers both Kodagu (75.6E) and Wayanad (76.1E)


@dataclass(frozen=True)
class Event:
    name: str
    date: str                          # event date (ISO)
    bbox: tuple[float, float, float, float]   # lon_min, lat_min, lon_max, lat_max
    pre: tuple[str, str]               # composite window before the event
    post: tuple[str, str]              # composite window after the event
    label_source: str


# Pre and post windows are the *same dry-season months* (Dec-Mar) one year
# apart, so phenology is matched and monsoon cloud is avoided.
EVENTS: dict[str, Event] = {
    "kodagu": Event(
        name="kodagu",
        date="2018-08-16",
        # inventory extent (75.462-75.833E, 12.386-12.690N) + ~1 km pad
        bbox=(75.452, 12.376, 75.843, 12.700),
        pre=("2017-12-01", "2018-03-31"),
        post=("2018-12-01", "2019-03-31"),
        label_source="Achaiah et al. (2024) Kodagu 2018 inventory, 217 polygons (CC BY 4.0)",
    ),
    "wayanad": Event(
        name="wayanad",
        date="2024-07-30",
        bbox=(76.05, 11.40, 76.25, 11.56),
        pre=("2023-12-01", "2024-03-31"),
        post=("2024-12-01", "2025-03-31"),
        label_source="Seeded Sentinel-2 change reference mask (see landsentry/labels.py)",
    ),
}

KODAGU_INVENTORY = ROOT / "data" / "kodagu_inventory" / "Landslide_polygons.shp"
# Location of the Mundakkai-Chooralmala debris flow (point placemark,
# "Mundakkai Landslide 2024.kmz", as used by the original LandSentry repo).
WAYANAD_SEED_LONLAT = (76.13479, 11.46458)


def event_dir(name: str) -> Path:
    return DATA_DIR / name
