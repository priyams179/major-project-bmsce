#!/usr/bin/env python3
"""
Sentinel-1/2 and Copernicus DEM Downloader for India Landslide Events

Provides pre-configured profiles, automated Copernicus DEM retrieval from AWS,
and operational acquisition workflows for India's landslide hotspots:
  - Wayanad 2024 (11.51°N, 76.15°E) — Western Ghats, Kerala (Tile: 43PGP, Track: 63)
  - Chamoli 2021 (30.37°N, 79.73°E) — Himalayas, Uttarakhand (Tile: 44RMR, Track: 27)
  - Joshimath 2023 (30.55°N, 79.56°E) — Himalayas, Uttarakhand (Tile: 44RMR, Track: 129)
  - Irshalwadi 2023 (18.93°N, 73.23°E) — Western Ghats, Maharashtra (Tile: 43QCB, Track: 63)
  - Tupul 2022 (24.71°N, 93.64°E) — Northeast India, Manipur (Tile: 46RFS, Track: 143)
  - Pettimudi 2020 (10.17°N, 77.02°E) — Western Ghats, Kerala (Tile: 43PEP, Track: 63)

Usage:
    python download_sentinel.py --list-events
    python download_sentinel.py --event wayanad_2024 --dem-only --output data/raw/dem
    python download_sentinel.py --event chamoli_2021 --output data/raw/
"""

import argparse
import os
import sys
import urllib.request
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Pre-configured India Landslide Event Catalog
# ---------------------------------------------------------------------------
INDIA_LANDSLIDE_EVENTS: Dict[str, dict] = {
    "wayanad_2024": {
        "name": "Wayanad Debris Avalanche & Mud Torrent, Kerala",
        "date": "2024-07-30",
        "lat": 11.51,
        "lon": 76.15,
        "bbox": (75.90, 11.30, 76.40, 11.80),
        "sentinel2_tile": "43PGP",
        "sentinel1_track": 63,
        "utm_zone": "EPSG:32643",
        "dem_tiles": ["N11_E076"],
        "pre_date_range": ("2024-06-01", "2024-07-29"),
        "post_date_range": ("2024-07-31", "2024-09-30"),
        "trigger": "Extreme monsoonal precipitation (572 mm in 48h)",
        "fatalities": 400,
    },
    "chamoli_2021": {
        "name": "Chamoli Rock-Ice Avalanche & Flash Flood, Uttarakhand",
        "date": "2021-02-07",
        "lat": 30.37,
        "lon": 79.73,
        "bbox": (79.40, 30.10, 80.10, 30.70),
        "sentinel2_tile": "44RMR",
        "sentinel1_track": 27,
        "utm_zone": "EPSG:32644",
        "dem_tiles": ["N30_E079"],
        "pre_date_range": ("2021-01-01", "2021-02-06"),
        "post_date_range": ("2021-02-08", "2021-03-31"),
        "trigger": "Ronti Peak hanging glacier and rock wall detachment",
        "fatalities": 204,
    },
    "joshimath_2023": {
        "name": "Joshimath Town Ground Subsidence & Slope Creep, Uttarakhand",
        "date": "2023-01-01",
        "lat": 30.55,
        "lon": 79.56,
        "bbox": (79.30, 30.40, 79.80, 30.80),
        "sentinel2_tile": "44RMR",
        "sentinel1_track": 129,
        "utm_zone": "EPSG:32644",
        "dem_tiles": ["N30_E079"],
        "pre_date_range": ("2022-10-01", "2022-12-31"),
        "post_date_range": ("2023-01-10", "2023-03-31"),
        "trigger": "Paleo-landslide reactivation & pore-water saturation",
        "fatalities": 0,
    },
    "irshalwadi_2023": {
        "name": "Irshalwadi Debris Flow, Khalapur, Raigad, Maharashtra",
        "date": "2023-07-19",
        "lat": 18.93,
        "lon": 73.23,
        "bbox": (72.90, 18.60, 73.60, 19.30),
        "sentinel2_tile": "43QCB",
        "sentinel1_track": 63,
        "utm_zone": "EPSG:32643",
        "dem_tiles": ["N18_E073"],
        "pre_date_range": ("2023-05-01", "2023-07-18"),
        "post_date_range": ("2023-07-20", "2023-09-30"),
        "trigger": "Torrential monsoon rainfall (499 mm over 3 days)",
        "fatalities": 86,
    },
    "tupul_2022": {
        "name": "Tupul Railway Yard Debris Flow, Noney, Manipur",
        "date": "2022-06-29",
        "lat": 24.71,
        "lon": 93.64,
        "bbox": (93.30, 24.40, 94.10, 25.10),
        "sentinel2_tile": "46RFS",
        "sentinel1_track": 143,
        "utm_zone": "EPSG:32646",
        "dem_tiles": ["N24_E093"],
        "pre_date_range": ("2022-04-01", "2022-06-28"),
        "post_date_range": ("2022-06-30", "2022-09-30"),
        "trigger": "Slope cut destabilization & continuous heavy rainfall",
        "fatalities": 61,
    },
    "pettimudi_2020": {
        "name": "Pettimudi Debris Avalanche, Rajamala, Idukki, Kerala",
        "date": "2020-08-06",
        "lat": 10.17,
        "lon": 77.02,
        "bbox": (76.80, 10.00, 77.30, 10.50),
        "sentinel2_tile": "43PEP",
        "sentinel1_track": 63,
        "utm_zone": "EPSG:32643",
        "dem_tiles": ["N10_E077"],
        "pre_date_range": ("2020-05-01", "2020-08-05"),
        "post_date_range": ("2020-08-07", "2020-11-30"),
        "trigger": "Extreme monsoon downpour (>600 mm in 72h)",
        "fatalities": 70,
    },
}


# ---------------------------------------------------------------------------
# Copernicus DEM GLO-30 Download
# ---------------------------------------------------------------------------
def download_copernicus_dem_tile(tile_id: str, cache_dir: str = "data/dem_cache") -> str:
    """
    Download a 30-meter Copernicus DEM (GLO-30) tile from AWS Open Data.

    Parameters
    ----------
    tile_id : str
        Format 'N11_E076' or 'N30_E079'.
    cache_dir : str
        Local directory to save the downloaded GeoTIFF tile.

    Returns
    -------
    local_path : str
        Path to the saved GeoTIFF.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"Copernicus_DSM_COG_10_{tile_id}_00_DEM.tif")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        print(f"  [Cache hit] Copernicus DEM tile {tile_id} already exists at: {local_path}")
        return local_path

    url = (
        f"https://copernicus-dem-30m.s3.amazonaws.com/"
        f"Copernicus_DSM_COG_10_{tile_id}_00_DEM/"
        f"Copernicus_DSM_COG_10_{tile_id}_00_DEM.tif"
    )

    print(f"  Downloading Copernicus DEM tile {tile_id} from AWS Open Data...")
    print(f"  URL: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LandslideDetection/1.0"})
        tmp_path = local_path + ".tmp"
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 65536
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r    Progress: {downloaded / 1e6:.2f} MB / {total_size / 1e6:.2f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
            print()
        os.replace(tmp_path, local_path)
        print(f"  Saved DEM tile to: {local_path} ({os.path.getsize(local_path) / 1e6:.2f} MB)")
        return local_path
    except Exception as e:
        if os.path.exists(local_path + ".tmp"):
            os.remove(local_path + ".tmp")
        print(f"  [Warning] Direct AWS Copernicus DEM download failed: {e}")
        print("  Attempting fallback to Mapzen/AWS Terrain Tiles...")
        return _fallback_terrain_tile(tile_id, cache_dir)


def _fallback_terrain_tile(tile_id: str, cache_dir: str) -> str:
    """Fallback fetcher using AWS elevation tiles."""
    parts = tile_id.split("_")
    lat = float(parts[0][1:]) if parts[0].startswith("N") else -float(parts[0][1:])
    lon = float(parts[1][1:]) if parts[1].startswith("E") else -float(parts[1][1:])
    import math

    lat_rad = math.radians(lat + 0.5)
    zoom = 10
    n = 2.0 ** zoom
    xtile = int((lon + 0.5 + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{zoom}/{xtile}/{ytile}.tif"
    local_path = os.path.join(cache_dir, f"dem_fallback_{tile_id}.tif")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        print(f"  Saved fallback DEM tile to: {local_path}")
        return local_path
    except Exception as err:
        print(f"  [Error] Fallback DEM download failed: {err}")
        return ""


# ---------------------------------------------------------------------------
# Instructions and Query Builders
# ---------------------------------------------------------------------------
def print_copernicus_dataspace_instructions(event_key: str) -> None:
    """Print complete acquisition queries for Copernicus Data Space Ecosystem."""
    if event_key not in INDIA_LANDSLIDE_EVENTS:
        print(f"Unknown event key '{event_key}'")
        return

    ev = INDIA_LANDSLIDE_EVENTS[event_key]
    min_lon, min_lat, max_lon, max_lat = ev["bbox"]

    print("=" * 80)
    print(f"COPERNICUS DATA SPACE ACQUISITION GUIDE: {ev['name']}")
    print("=" * 80)
    print(f"Date of Event:      {ev['date']}")
    print(f"Coordinates:        {ev['lat']}°N, {ev['lon']}°E")
    print(f"Bounding Box:       [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
    print(f"Sentinel-2 MGRS:    {ev['sentinel2_tile']}")
    print(f"Sentinel-1 Track:   Relative Orbit {ev['sentinel1_track']}")
    print(f"UTM Zone:           {ev['utm_zone']}")
    print("-" * 80)
    print("\n1. SENTINEL-1 SAR (MISSION-CRITICAL FOR MONSOON CLOUD COVER):")
    print("   Portal: https://dataspace.copernicus.eu (Free registration)")
    print("   OData Search Query:")
    print(
        f"   https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
        f"$filter=Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))') and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD') and "
        f"ContentDate/Start gt {ev['pre_date_range'][0]}T00:00:00.000Z and "
        f"ContentDate/Start lt {ev['post_date_range'][1]}T23:59:59.000Z"
    )
    print("\n2. SENTINEL-2 OPTICAL (POST-MONSOON CLOUD-FREE SCENES):")
    print(
        f"   Search criteria: Collection 'SENTINEL-2', Level-2A (Bottom of Atmosphere)\n"
        f"   MGRS Tile: {ev['sentinel2_tile']}\n"
        f"   Pre-event window:  {ev['pre_date_range'][0]} to {ev['pre_date_range'][1]}\n"
        f"   Post-event window: {ev['post_date_range'][0]} to {ev['post_date_range'][1]}\n"
        f"   Cloud cover filter: < 20%"
    )
    print("\n3. PYTHON API ACQUISITION SCRIPT (sentinelsat / requests):")
    print(f"""
from sentinelsat import SentinelAPI

api = SentinelAPI('YOUR_USERNAME', 'YOUR_PASSWORD', 'https://dataspace.copernicus.eu/resto/api')
footprint = 'POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))'

# Sentinel-1 SAR products
products = api.query(
    footprint,
    date=('{ev["pre_date_range"][0].replace("-", "")}', '{ev["post_date_range"][1].replace("-", "")}'),
    platformname='Sentinel-1',
    producttype='GRD',
    sensoroperationalmode='IW'
)
print(f"Found {{len(products)}} Sentinel-1 acquisitions.")
api.download_all(products, directory_path='data/raw/sentinel1')
""")
    print("=" * 80)


def list_available_events() -> None:
    """Print a summary table of all pre-configured India events."""
    print("=" * 95)
    print(f"{'Key':<16} | {'Date':<10} | {'S2 Tile':<7} | {'S1 Trk':<6} | {'DEM Tile':<8} | {'Event Description'}")
    print("-" * 95)
    for k, v in INDIA_LANDSLIDE_EVENTS.items():
        print(f"{k:<16} | {v['date']:<10} | {v['sentinel2_tile']:<7} | {v['sentinel1_track']:<6} | {v['dem_tiles'][0]:<8} | {v['name']}")
    print("=" * 95)


def download_event_data(event_key: str, output_dir: str = "data/raw", include_dem: bool = True) -> Dict[str, List[str]]:
    """Download available data (DEM tiles and instructions) for an event."""
    if event_key not in INDIA_LANDSLIDE_EVENTS:
        raise ValueError(f"Unknown event key '{event_key}'. Available: {list(INDIA_LANDSLIDE_EVENTS.keys())}")

    ev = INDIA_LANDSLIDE_EVENTS[event_key]
    res: Dict[str, List[str]] = {"dem": [], "event_key": [event_key]}

    dem_cache = os.path.join(output_dir, "dem")
    if include_dem:
        print(f"\n--- Fetching Copernicus DEM GLO-30 for {ev['name']} ---")
        for tile in ev["dem_tiles"]:
            path = download_copernicus_dem_tile(tile, cache_dir=dem_cache)
            if path:
                res["dem"].append(path)

    print_copernicus_dataspace_instructions(event_key)
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download Satellite & Elevation Data for Indian Landslide Events")
    parser.add_argument("--list-events", action="store_true", help="List all pre-configured Indian landslide events")
    parser.add_argument("--event", choices=list(INDIA_LANDSLIDE_EVENTS.keys()), help="Target landslide event identifier")
    parser.add_argument("--dem-only", action="store_true", help="Only download Copernicus GLO-30 DEM tiles")
    parser.add_argument("--output", default="data/raw", help="Output directory for downloaded assets")
    args = parser.parse_args()

    if args.list_events:
        list_available_events()
        return

    if not args.event:
        print("Please specify an event with --event or run with --list-events.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    download_event_data(args.event, output_dir=args.output, include_dem=True)


if __name__ == "__main__":
    main()

