
"""
Global Landsat 8/9 infrared landslide imagery collector.

Input:
    landslide_events.csv
        event_id,event_date,latitude,longitude

Output:
    Google Drive / Landslide_Landsat_Global/
        <event_id>_PRE.tif
        <event_id>_POST.tif

Each GeoTIFF contains:
    nir          - Landsat Band 5 surface reflectance
    swir1        - Landsat Band 6 surface reflectance
    swir2        - Landsat Band 7 surface reflectance
    valid_obs    - Number of valid observations contributing
                   to the composite

NOTE:
    This downloads imagery, NOT landslide segmentation masks.
"""

import csv
import math
import time
from datetime import datetime, timedelta

import ee


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "major-project-512310"
EVENTS_CSV = "landslide_events.csv"
DRIVE_FOLDER = "Landslide_Landsat_Global_v2"

# 10 km x 10 km area around each reported event.
# Increase this to 15 or 20 km for larger landslides.
RADIUS_KM = 5

# Longer windows increase the chance of cloud-free observations,
# but can include unrelated changes.
PRE_DAYS = 180
POST_DAYS = 180

# Start with manageable batches.
MAX_EVENTS_PER_RUN = 10
OFFSET = 30

# Do not discard partially cloudy scenes using scene-level
# cloud cover: QA_PIXEL will mask contaminated pixels.
MIN_SCENES_PER_PERIOD = 1

# Collection 2 Level-2 surface-reflectance collections.
LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_9 = "LANDSAT/LC09/C02/T1_L2"

BAND_NAMES = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
OUTPUT_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

NODATA = -9999


# ============================================================
# INITIALIZATION
# ============================================================

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)


# ============================================================
# EVENT CSV
# ============================================================

def load_events(path):
    events = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {
            "event_id", "event_date", "latitude", "longitude"
        }

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)}"
            )

        for row in reader:
            try:
                event_date = datetime.strptime(
                    row["event_date"].strip(), "%Y-%m-%d"
                ).date()

                lat = float(row["latitude"])
                lon = float(row["longitude"])
                event_id = row["event_id"].strip()

                if not event_id:
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue

                # Landsat 8 begins in 2013.
                # Future or pre-Landsat-8 events are not suitable.
                if event_date < datetime(2013, 4, 1).date():
                    continue

                if event_date > datetime.utcnow().date():
                    continue

                events.append({
                    "event_id": event_id,
                    "date": event_date,
                    "lat": lat,
                    "lon": lon,
                })

            except (ValueError, TypeError):
                print("Skipping invalid CSV row:", row)

    # Remove duplicate IDs.
    unique = {}
    for event in events:
        unique[event["event_id"]] = event

    return list(unique.values())


# ============================================================
# GEOMETRY
# ============================================================

def event_region(lat, lon, radius_km):
    """Approximate square around event; suitable for initial crops."""

    lat_delta = radius_km / 111.32
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    lon_delta = radius_km / (111.32 * cos_lat)

    return ee.Geometry.Rectangle([
        lon - lon_delta,
        lat - lat_delta,
        lon + lon_delta,
        lat + lat_delta,
    ])


# ============================================================
# MASK AND SCALE
# ============================================================

def prepare_landsat(image):
    """
    Apply Collection 2 QA masks and surface-reflectance scaling.
    Output reflectance is scaled to physical units.
    """

    qa = image.select("QA_PIXEL")

    # Bits 0-5: fill, dilated cloud, cirrus, cloud,
    # cloud shadow, and snow.
    bad_pixel_bits = (
        (1 << 0)
        | (1 << 1)
        | (1 << 2)
        | (1 << 3)
        | (1 << 4)
        | (1 << 5)
    )

    clear = qa.bitwiseAnd(bad_pixel_bits).eq(0)

    # Reject saturation in the selected bands.
    radsat = image.select("QA_RADSAT")
    saturated_bits = (1 << 4) | (1 << 5) | (1 << 6)
    unsaturated = radsat.bitwiseAnd(saturated_bits).eq(0)

    reflectance = (
        image.select(BAND_NAMES, OUTPUT_NAMES)
        .multiply(0.0000275)
        .add(-0.2)
    )

    # Keep the same pixel valid only when all selected bands
    # are clear and unsaturated.
    return (
        reflectance
        .updateMask(clear)
        .updateMask(unsaturated)
        .copyProperties(
            image,
            ["system:time_start", "LANDSAT_PRODUCT_ID"]
        )
    )


# ============================================================
# COLLECTION SEARCH
# ============================================================

def get_collection(region, start_date, end_date):
    """Get all intersecting Landsat 8 and 9 scenes in a period."""

    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")

    l8 = (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(region)
        .filterDate(start, end)
    )

    l9 = (
        ee.ImageCollection(LANDSAT_9)
        .filterBounds(region)
        .filterDate(start, end)
    )

    return l8.merge(l9).map(prepare_landsat)


# ============================================================
# EXPORT
# ============================================================

def export_period(event, region, start, end, period):
    """Create one composite and submit a Drive export task."""

    collection = get_collection(region, start, end)
    scene_count = collection.size().getInfo()

    print(
        f"  {period}: {start} to {end}; "
        f"{scene_count} intersecting scenes"
    )

    if scene_count < MIN_SCENES_PER_PERIOD:
        print("  Skipping: no scenes in this period.")
        return False

    # Median of clear observations.
    composite = collection.median()

    # Count valid observations for each pixel.
    valid_count = (
        collection.select("nir")
        .count()
        .rename("valid_obs")
    )

    # Static slope (degrees) from SRTM; resampled to Landsat grid.
    slope = (
        ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
        .resample("bilinear")
        .rename("slope")
    )

    output = (
        composite
        .addBands(valid_count)
        .addBands(slope)
        .select([
            "blue", "green", "red", "nir", "swir1", "swir2",
            "slope", "valid_obs",
        ])
        .toFloat()
        .unmask(NODATA, False)
        .clip(region)
    )

    description = f"{event['event_id']}_{period}"

    task = ee.batch.Export.image.toDrive(
        image=output,
        description=description,
        folder=DRIVE_FOLDER,
        fileNamePrefix=description,
        region=region,
        scale=30,
        maxPixels=1_000_000_000,
        fileFormat="GeoTIFF",
        formatOptions={"noData": NODATA},
    )

    task.start()
    print("  Submitted:", description, "| task:", task.id)

    return True


def process_event(event):
    """Submit pre-event and post-event exports."""

    event_id = event["event_id"]
    event_date = event["date"]

    print(
        f"\nProcessing {event_id}: "
        f"{event_date} at ({event['lat']}, {event['lon']})"
    )

    region = event_region(
        event["lat"], event["lon"], RADIUS_KM
    )

    pre_start = event_date - timedelta(days=PRE_DAYS)
    pre_end = event_date

    post_start = event_date
    post_end = event_date + timedelta(days=POST_DAYS)

    pre_ok = export_period(
        event, region, pre_start, pre_end, "PRE"
    )

    # Brief pause between task submissions.
    time.sleep(1)

    post_ok = export_period(
        event, region, post_start, post_end, "POST"
    )

    return pre_ok, post_ok


# ============================================================
# MAIN
# ============================================================

def main():
    events = load_events(EVENTS_CSV)

    if not events:
        raise RuntimeError(
            "No valid events found. Check the input CSV."
        )

    print("Total valid events:", len(events))

    selected = events[
        OFFSET: OFFSET + MAX_EVENTS_PER_RUN
    ]

    if not selected:
        print("No events in this batch. Increase or reset OFFSET.")
        return

    print("Events in this batch:", len(selected))
    print("Expected maximum export tasks:", len(selected) * 2)

    submitted = 0
    skipped = 0

    for event in selected:
        try:
            before, after = process_event(event)
            submitted += int(before) + int(after)
            skipped += int(not before) + int(not after)

        except Exception as exc:
            print(
                f"ERROR on event {event['event_id']}: {exc}"
            )

    print("\nBatch complete.")
    print("Export tasks submitted:", submitted)
    print("Periods skipped:", skipped)
    print(
        f"Next batch: set OFFSET = "
        f"{OFFSET + MAX_EVENTS_PER_RUN}"
    )
    print(
        "Monitor task states in the Earth Engine Tasks page "
        "and check Google Drive after completion."
    )


if __name__ == "__main__":
    main()

