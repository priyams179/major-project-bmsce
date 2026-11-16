"""
Step 1 — Build conditioning-factor raster using Google Earth Engine.

Exports a 30-m 6-band GeoTIFF for Rudraprayag district:
  Band 1: elevation (m), from SRTM
  Band 2: slope (degrees)
  Band 3: aspect (degrees)
  Band 4: NDVI (Sentinel-2 median, Oct 2024–Mar 2025)
  Band 5: land cover (ESA WorldCover, integer class)
  Band 6: distance to nearest free-flowing river (m)

Usage:
    python src/features.py --project <gee_project_id> [--drive-folder landslide_project]

The export runs as a GEE task (5–20 min) and saves 'rudraprayag_features.tif'
to your Google Drive folder. Download it and place it in data/.
"""
from __future__ import annotations

import argparse
import sys


def build_features(project_id: str, drive_folder: str = "landslide_project") -> None:
    """Submit GEE export task for the Rudraprayag feature stack."""
    try:
        import ee
    except ImportError:
        sys.exit("earthengine-api not installed. Run: uv add earthengine-api")

    print(f"Authenticating with GEE project: {project_id}")
    ee.Authenticate()
    ee.Initialize(project=project_id)

    # ── Study area ──────────────────────────────────────────────────────────
    districts = ee.FeatureCollection("FAO/GAUL/2025/level2")
    match = (
        districts
        .filter(ee.Filter.eq("ISO3_CODE", "IND"))
        .filter(ee.Filter.stringContains("GAUL2_NAME", "Rudra"))
    )
    n = match.size().getInfo()
    if n > 0:
        aoi = match.geometry()
        print(f"Using official Rudraprayag boundary (GAUL 2025 — {n} polygon(s) found)")
    else:
        # Approximate bounding box if GAUL lookup fails
        aoi = ee.Geometry.Rectangle([78.7, 30.1, 79.5, 30.9])
        print("GAUL lookup returned 0 matches — using approximate bounding box")

    # ── Terrain (SRTM 30 m) ─────────────────────────────────────────────────
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)

    # ── NDVI — Sentinel-2 SR median, Oct 2024–Mar 2025, <20% cloud ──────────
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate("2024-10-01", "2025-04-01")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"])).median().rename("ndvi")

    # ── Land cover (ESA WorldCover v200) ────────────────────────────────────
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")

    # ── Distance to free-flowing rivers (WWF HydroSHEDS) ───────────────────
    rivers = (
        ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
        .filterBounds(aoi.buffer(20_000))
    )
    dist_river = (
        ee.Image()
        .paint(rivers, 1)
        .distance(ee.Kernel.euclidean(radius=20_000, units="meters"))
        .rename("dist_river")
    )

    # ── Stack and clip ───────────────────────────────────────────────────────
    features = (
        ee.Image.cat([elevation, slope, aspect, ndvi, landcover, dist_river])
        .clip(aoi)
    )

    # ── Export ───────────────────────────────────────────────────────────────
    task = ee.batch.Export.image.toDrive(
        image=features.toFloat(),
        description="rudraprayag_features",
        folder=drive_folder,
        region=aoi,
        scale=30,
        crs="EPSG:32644",   # UTM zone 44N — covers Uttarakhand, metres
        maxPixels=1e10,
    )
    task.start()
    print(
        f"\nGEE export task started — id: {task.id}\n"
        f"Output: Google Drive / {drive_folder} / rudraprayag_features.tif\n"
        "Typical wait: 5–20 minutes.\n"
        "After it completes, download the file and place it in data/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Rudraprayag feature raster via GEE")
    parser.add_argument("--project", required=True, help="Google Earth Engine project ID")
    parser.add_argument("--drive-folder", default="landslide_project",
                        help="Google Drive folder for export (default: landslide_project)")
    args = parser.parse_args()
    build_features(project_id=args.project, drive_folder=args.drive_folder)


if __name__ == "__main__":
    main()

