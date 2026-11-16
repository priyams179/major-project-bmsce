"""Download 30 m conditioning factor GeoTIFF stack for Uttarakhand (Rudraprayag)
from Google Earth Engine.

Factors exported (6 bands):
  1. elevation   - NASA SRTM (USGS/SRTMGL1_003) (m)
  2. slope       - terrain slope derived from SRTM (degrees)
  3. aspect      - terrain aspect derived from SRTM (degrees)
  4. ndvi        - Sentinel-2 SR median composite, Oct 2024–Mar 2025
  5. landcover   - ESA WorldCover v200 (10 m, resampled)
  6. dist_river  - distance to WWF HydroSHEDS free-flowing river (m)

Output:
  rudraprayag_features.tif in data/Uttarakhand_Rudraprayag/ (via Google Drive export)

Usage:
  python download_uttarakhand_features.py --project <your-ee-project>
  python download_uttarakhand_features.py --project <your-ee-project> --drive-folder landslide_project
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUT_DIR = Path("data/Uttarakhand_Rudraprayag")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Earth Engine project ID")
    ap.add_argument("--drive-folder", default="landslide_project",
                    help="Google Drive target folder (default: landslide_project)")
    ap.add_argument("--out", default=str(OUT_DIR), help="Local output directory")
    args = ap.parse_args()

    import ee
    print(f"Authenticating with GEE project: {args.project}...")
    ee.Authenticate()
    ee.Initialize(project=args.project)

    districts = ee.FeatureCollection("FAO/GAUL/2025/level2")
    match = (districts
             .filter(ee.Filter.eq("ISO3_CODE", "IND"))
             .filter(ee.Filter.stringContains("GAUL2_NAME", "Rudra")))

    if match.size().getInfo() > 0:
        aoi = match.geometry()
        print("Using official FAO GAUL 2025 Rudraprayag boundary")
    else:
        aoi = ee.Geometry.Rectangle([78.7, 30.1, 79.5, 30.9])
        print("GAUL boundary not found, using bounding box [78.7, 30.1, 79.5, 30.9]")

    # SRTM 30 m
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = srtm.select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)

    # Sentinel-2 NDVI median (Oct 2024 - Mar 2025, <20% cloud)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate("2024-10-01", "2025-04-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
    ndvi = s2.map(lambda img: img.normalizedDifference(["B8", "B4"])).median().rename("ndvi")

    # ESA WorldCover v200
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")

    # WWF HydroSHEDS free-flowing rivers
    rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers").filterBounds(aoi.buffer(20000))
    dist_river = ee.Image().paint(rivers, 1).distance(ee.Kernel.euclidean(20000, "meters")).rename("dist_river")

    features = ee.Image.cat([elevation, slope, aspect, ndvi, landcover, dist_river]).clip(aoi)

    task = ee.batch.Export.image.toDrive(
        image=features.toFloat(),
        description="rudraprayag_features",
        folder=args.drive_folder,
        region=aoi,
        scale=30,
        crs="EPSG:32644",
        maxPixels=1e10,
    )
    task.start()
    print(f"Task started: {task.id}")
    print(f"Check status in GEE Tasks tab. When complete, copy rudraprayag_features.tif into {args.out}/.")


if __name__ == "__main__":
    main()

