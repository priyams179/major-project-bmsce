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
import json
import os
import re
import shutil
import sys
import urllib.parse
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

    # The bucket keys use degree+minute form: N11_E076 -> N11_00_E076_00.
    # Building the key straight from tile_id 404s and silently degrades to
    # the Mapzen fallback below, which is not the Copernicus GLO-30 DEM.
    parts = tile_id.split("_")
    key_tile = f"{parts[0]}_00_{parts[1]}_00" if len(parts) == 2 else tile_id
    url = (
        f"https://copernicus-dem-30m.s3.amazonaws.com/"
        f"Copernicus_DSM_COG_10_{key_tile}_DEM/"
        f"Copernicus_DSM_COG_10_{key_tile}_DEM.tif"
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
# Sentinel-1 GRD Download (CDSE OData + download REST API)
# ---------------------------------------------------------------------------
CDSE_odata = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_auth = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_download = "https://download.dataspace.copernicus.eu/odata/v1/Products"

# ---------------------------------------------------------------------------
# Keyless mirror: Element84 sentinel-s1-l1c open-data bucket
# ---------------------------------------------------------------------------
S1_GRD_MIRROR = "https://sentinel-s1-l1c.s3.amazonaws.com/"


def _s3_list_keys(prefix: str) -> List[str]:
    """
    List every object key under *prefix* in the anonymous mirror bucket.

    Paginates through ListObjectsV2 results: a single day prefix holds more
    than 1000 keys, and the unpaginated first page silently omits products
    whose folders sort late in the day.
    """
    keys: List[str] = []
    query = f"list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}"
    while query:
        req = urllib.request.Request(
            f"{S1_GRD_MIRROR}?{query}",
            headers={"User-Agent": "LandslideAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", xml))
        token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        query = (
            f"list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}"
            f"&continuation-token={urllib.parse.quote(token.group(1))}"
            if token else None
        )
    return keys


def _s1_mirror_prefixes(product_name: str) -> List[str]:
    """
    Candidate ``GRD/{y}/{m}/{d}/{mode}/{pol}/`` prefixes for a product name.

    Both padded and unpadded date forms are returned (the bucket mixes
    neither, but the convention is unpadded); the layout mirrors the SAFE
    tree, e.g. ``GRD/2024/7/25/IW/DV/S1A_IW_GRDH_1SDV_20240725T004900_…``.
    """
    base = product_name[:-5] if product_name.endswith(".SAFE") else product_name
    m = re.search(r"(\d{4})(\d{2})(\d{2})T", base)
    if not m:
        return []
    year, month, day = m.group(1), m.group(2), m.group(3)
    parts = base.split("_")
    mode = parts[1] if len(parts) > 1 else "IW"
    pol_code = parts[3] if len(parts) > 3 else ""
    pol = pol_code[-2:] if len(pol_code) >= 2 else "DV"
    prefixes = [
        f"GRD/{int(year)}/{int(month)}/{int(day)}/{mode}/{pol}/",
        f"GRD/{year}/{month}/{day}/{mode}/{pol}/",
    ]
    return list(dict.fromkeys(prefixes))


def _fetch_mirror_file(url: str, dest: str) -> bool:
    """Stream *url* to *dest* atomically, printing progress.  True on success."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "LandslideAI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total:
                    sys.stdout.write(
                        f"\r    {got / 1e6:.1f} / {total / 1e6:.1f} MB "
                        f"({100.0 * got / total:.0f}%)"
                    )
                    sys.stdout.flush()
        if total:
            sys.stdout.write("\n")
        os.replace(tmp, dest)
        return True
    except Exception as exc:                      # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"\n    [warn] download failed: {exc}")
        return False


def download_sentinel1_safe(product_name: str, dest_dir: str) -> Optional[str]:
    """
    Assemble a GRD ``.SAFE`` directory from the keyless mirror.

    Fetches the members ``preprocess_sar.py`` reads: ``manifest.safe``,
    ``measurement/*.tiff`` and ``annotation/*.xml`` (calibration and noise
    LUTs included).  No credentials are required.

    Parameters
    ----------
    product_name : str
        CDSE product name, e.g. ``S1A_IW_GRDH_1SDV_..._XXXX.SAFE``.
    dest_dir : str
        Directory the ``.SAFE`` folder is created in.

    Returns
    -------
    str or None
        Path of the assembled ``.SAFE`` directory, or None on failure.
        Already-fetched members are kept, so a re-run resumes.
    """
    base = product_name[:-5] if product_name.endswith(".SAFE") else product_name
    keys: List[str] = []
    root = ""
    for prefix in _s1_mirror_prefixes(product_name):
        try:
            listed = _s3_list_keys(prefix)
        except Exception as exc:                  # noqa: BLE001
            print(f"  [warn] mirror listing failed for {prefix}: {exc}")
            continue
        for key in listed:
            rest = key[len(prefix):]
            folder = rest.split("/", 1)[0]
            if folder.startswith(base):
                keys = [k for k in listed if k.startswith(prefix + folder)]
                root = f"{prefix}{folder}/"
                break
        if keys:
            break
    if not keys:
        print(f"  [error] {base} not found in the {S1_GRD_MIRROR} mirror.")
        return None

    members = []
    for key in keys:
        rel = key[len(root):]
        if rel == "manifest.safe" or rel.startswith("measurement/") or (
            rel.startswith("annotation/") and rel.endswith(".xml")
            and "/rfi/" not in rel
        ):
            members.append((rel, key))

    safe_dir = os.path.join(dest_dir, base + ".SAFE")
    print(f"  Mirror: {len(members)} members -> {safe_dir}")
    done = 0
    for rel, key in sorted(members):
        dest = os.path.join(safe_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"    [cache] {rel}")
            done += 1
            continue
        print(f"    {rel}")
        if _fetch_mirror_file(S1_GRD_MIRROR + urllib.parse.quote(key), dest):
            done += 1

    if done != len(members):
        print(f"  [warn] {done}/{len(members)} mirror members retrieved.")
        return None
    return safe_dir


def _cdse_token(client_id: str, client_secret: str) -> str:
    """Exchange CDSE client credentials for a bearer token."""
    payload = (
        f"client_id={client_id}&client_secret={client_secret}"
        f"&grant_type=client_credentials"
    ).encode("utf-8")
    req = urllib.request.Request(
        CDSE_auth,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]


def search_sentinel1_scenes(
    bbox: tuple,
    start_date: str,
    end_date: str,
    track: int | None = None,
) -> list:
    """
    Query CDSE for Sentinel-1 GRD products intersecting *bbox*.

    Landslides are detected from ground deformation, so the pre/post pair must
    share a relative orbit: a different track looks at the slope from a
    different incidence angle and the difference is dominated by that geometry
    rather than by the event.

    Parameters
    ----------
    bbox : tuple of float
        (lon_min, lat_min, lon_max, lat_max), WGS84.
    start_date, end_date : str
        Inclusive ISO dates, 'YYYY-MM-DD'.
    track : int, optional
        Sentinel-1 relative orbit number.

    Returns
    -------
    list of dict
        Product records containing 'Id', 'Name', and 'S3Path'.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    poly = (
        f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
    )
    query = (
        f"$filter=Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
        f"Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' "
        f"and a/OData.CSC.StringAttribute/Value eq 'GRD') and "
        f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
        f"ContentDate/Start lt {end_date}T23:59:59.000Z"
    )
    if track is not None:
        # Filter server-side: the relative orbit is an IntegerAttribute of
        # the product.  Product names only carry the *absolute* orbit
        # (e.g. _054910_), so a name-based needle for the track number can
        # never match and used to discard every candidate.
        query += (
            " and Attributes/OData.CSC.IntegerAttribute/any("
            "a:a/Name eq 'relativeOrbitNumber' and "
            f"a/OData.CSC.IntegerAttribute/Value eq {int(track)})"
        )
    url = f"{CDSE_odata}?$top=200&$orderby=ContentDate/Start asc&{query}"
    # http.client rejects raw spaces in the request target (Python >= 3.12
    # validates the path), so percent-encode them; curl does this silently,
    # which is why the query works outside Python.
    url = url.replace(" ", "%20")

    req = urllib.request.Request(url, headers={"User-Agent": "LandslideAI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())

    products = payload.get("value", [])

    return products


def download_sentinel1_pair(
    event_key: str,
    output_dir: str = "data/raw",
    client_id: str | None = None,
    client_secret: str | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Download the best (closest-in-time, lowest-cloud-equivalent) S1 pre/post
    GRD pair for one catalogued event.

    Credentials come from ``CDSE_CLIENT_ID`` / ``CDSE_CLIENT_SECRET``, or from
    the ``client_id`` / ``client_secret`` arguments.  Without them the
    official CDSE zip route is skipped and each product is assembled from
    the anonymous ``sentinel-s1-l1c`` open-data mirror instead, so no key is
    required.  A failed token exchange or zip download falls back to the
    same mirror.

    Parameters
    ----------
    event_key : str
        Key in ``INDIA_LANDSLIDE_EVENTS``.
    output_dir : str
        Root raw directory; products land in ``{output_dir}/sentinel1/{event_key}/``.
    client_id, client_secret : str, optional
        CDSE OAuth2 credentials.
    dry_run : bool
        List products without downloading.

    Returns
    -------
    dict with keys 'pre', 'post', 'files'.
    """
    if event_key not in INDIA_LANDSLIDE_EVENTS:
        raise ValueError(f"Unknown event '{event_key}'.")

    ev = INDIA_LANDSLIDE_EVENTS[event_key]
    pre_start, pre_end = ev["pre_date_range"]
    post_start, post_end = ev["post_date_range"]
    track = ev["sentinel1_track"]

    print(f"\n{'=' * 78}")
    print(f"Sentinel-1 GRD pair: {ev['name']}  ({ev['date']})")
    print(f"  relative orbit: {track}")
    print(f"  pre  window: {pre_start} → {pre_end}")
    print(f"  post window: {post_start} → {post_end}")
    print(f"{'=' * 78}")

    pre_products = search_sentinel1_scenes(
        ev["bbox"], pre_start, pre_end, track=track
    )
    post_products = search_sentinel1_scenes(
        ev["bbox"], post_start, post_end, track=track
    )

    print(f"\n  pre candidates : {len(pre_products)}")
    print(f"  post candidates: {len(post_products)}")

    result = {"event": event_key, "files": {}, "pre": None, "post": None}

    if not pre_products or not post_products:
        print("  [warn] Missing a full pre or post pair on the configured "
              "orbit.  Check the track number in the event profile.")
        return result

    # Closest to the event date on each side gives the tightest change window.
    pre_p = pre_products[-1]
    post_p = post_products[0]
    result["pre"], result["post"] = pre_p, post_p

    print(f"\n  PRE : {pre_p.get('Name')}")
    print(f"  POST: {post_p.get('Name')}")

    if dry_run:
        print("\n  --dry-run: not downloading.")
        for label, prod in (("pre", pre_p), ("post", post_p)):
            print(f"  {label}: {prod.get('S3Path') or prod.get('Id')}")
        return result

    cid = client_id or os.environ.get("CDSE_CLIENT_ID", "")
    secret = client_secret or os.environ.get("CDSE_CLIENT_SECRET", "")

    token = None
    if cid and secret:
        try:
            token = _cdse_token(cid, secret)
        except Exception as exc:                  # noqa: BLE001
            print(f"\n  [warn] CDSE token exchange failed ({exc}); "
                  "falling back to the keyless mirror.")
    else:
        print("\n  No CDSE credentials — using the keyless "
              "sentinel-s1-l1c mirror.")

    headers = {"Authorization": f"Bearer {token}"} if token else None

    for label, prod in (("pre", pre_p), ("post", post_p)):
        dest_dir = os.path.join(output_dir, "sentinel1", event_key, label)
        os.makedirs(dest_dir, exist_ok=True)

        if token:
            url = f"{CDSE_download}(Id eq '{prod['Id']}')/$value"
            fname = os.path.join(dest_dir, prod["Name"] + ".zip")

            if os.path.exists(fname) and os.path.getsize(fname) > 1024:
                print(f"  [cache] {fname}")
                result["files"][label] = dest_dir
                continue

            print(f"  Downloading {label}: {prod['Name']}")
            req = urllib.request.Request(url, headers=headers)
            tmp = fname + ".tmp"
            try:
                with urllib.request.urlopen(req, timeout=300) as resp, \
                        open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp, fh, length=1 << 20)
                os.replace(tmp, fname)
                result["files"][label] = dest_dir
                continue
            except Exception as exc:              # noqa: BLE001
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"    [error] {label} CDSE download failed: {exc}")
                print("    falling back to the keyless mirror.")

        safe = download_sentinel1_safe(prod["Name"], dest_dir)
        if safe:
            result["files"][label] = safe

    return result


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
    parser.add_argument(
        "--sar", action="store_true",
        help="Download the Sentinel-1 pre/post GRD pair instead of just DEM + instructions.")
    parser.add_argument(
        "--sar-dry-run", action="store_true",
        help="List the Sentinel-1 pre/post products without downloading.")
    args = parser.parse_args()

    if args.list_events:
        list_available_events()
        return

    if not args.event:
        print("Please specify an event with --event or run with --list-events.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if args.sar or args.sar_dry_run:
        download_sentinel1_pair(
            args.event,
            output_dir=args.output,
            dry_run=args.sar_dry_run,
        )
        return

    download_event_data(args.event, output_dir=args.output, include_dem=True)


if __name__ == "__main__":
    main()

