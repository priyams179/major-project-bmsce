#!/usr/bin/env python3
"""
Landsat Collection 2 Level-2 Downloader for India Landslide Events

Replaces Sentinel-2 in the optical slot.  Landsat-8/9 surface reflectance is
30 m, has a 16-day revisit per satellite (8 days combined), and is distributed
by USGS M2M.  It is still cloud-prone during the monsoon, but the shorter
revisit makes a usable pre/post pair far more likely than with Sentinel-2
alone.

Authentication
--------------
No key is required by default: searches and downloads use the anonymous
Microsoft Planetary Computer mirror of Landsat Collection 2 Level-2 when
``USGS_M2M_API_KEY`` is unset.  Supply a key (or ``--backend m2m``) to use
the official USGS EROS route instead:

    export USGS_M2M_API_KEY='your_key_here'

``--backend auto`` (the default) picks M2M when the key is set and the
keyless mirror otherwise.

Usage
-----
    # List every Landsat scene covering Wayanad in the pre-event window
    python download_landsat.py --event wayanad_2024 --search-only

    # Download the best (lowest cloud) pre and post scene for an event
    python download_landsat.py --event wayanad_2024 --max-cloud 20

    # Print scenes for every catalogued event
    python download_landsat.py --all-events --search-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# USGS M2M API
# ---------------------------------------------------------------------------
M2M_BASE = "https://m2m.cr.usgs.gov/api/api/json/stable"
EARTH_EXPLORER_BASE = "https://earthexplorer.usgs.gov"

# ---------------------------------------------------------------------------
# Keyless mirror: Microsoft Planetary Computer STAC
# ---------------------------------------------------------------------------
PC_STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS_TOKEN = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
)

#: STAC asset key -> band suffix written to disk.  ``preprocess_optical``
#: discovers bands by the ``_SR_B*`` / ``_QA_PIXEL`` suffix, and the mirror's
#: hrefs carry the original USGS filenames, so the two line up directly.
PC_BAND_ASSETS: Dict[str, str] = {
    "blue":     "SR_B2",
    "green":    "SR_B3",
    "red":      "SR_B4",
    "nir08":    "SR_B5",
    "swir16":   "SR_B6",
    "swir22":   "SR_B7",
    "qa_pixel": "QA_PIXEL",
}

#: Anonymous SAS token, fetched once per process (valid for ~24 h).
_PC_TOKEN: Optional[str] = None


def resolve_backend(backend: Optional[str] = None) -> str:
    """
    Pick the acquisition backend.

    ``auto`` (default) selects the official USGS M2M route when
    ``USGS_M2M_API_KEY`` is set and the keyless Planetary Computer
    mirror otherwise.
    """
    if backend and backend != "auto":
        return backend
    if os.environ.get("USGS_M2M_API_KEY", "").strip():
        return "m2m"
    return "pc"

#: Landsat sensors in chronological order, so L8 wins ties on true-colour.
DEFAULT_SENSORS = ("L8", "L9")

#: Product IDs for Collection 2 Level-2 surface reflectance.
L2_PRODUCT_IDS = (
    "LC08_L2SP_",
    "LC09_L2SP_",
    "LT08_L2SP_",
    "LT09_L2SP_",
    "LC08_L2SR_",
    "LC09_L2SR_",
)

#: Landsat-8/9 pre-configured windows, derived from the event date.  Landsat
#: has no global two-tier (Collection 1 vs 2) naming problem, so one window per
#: side is enough: a month before and three months after, which comfortably
#: spans the monsoon for every catalogued event.
def _event_windows(event_date: str) -> Tuple[str, str]:
    """Return (pre_start, post_end) ISO dates bracketing an event."""
    from datetime import date, timedelta

    y, m, d = (int(x) for x in event_date.split("-"))
    ev = date(y, m, d)
    pre_start = ev - timedelta(days=45)
    post_end = ev + timedelta(days=120)
    return pre_start.isoformat(), post_end.isoformat()


# ---------------------------------------------------------------------------
# Scene search
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("USGS_M2M_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "USGS_M2M_API_KEY is not set.  Run with --backend pc to use "
            "the keyless Planetary Computer mirror, or register at "
            "https://ers.cr.usgs.gov/ and export:\n"
            "    export USGS_M2M_API_KEY='your_key_here'"
        )
    return key


def _m2m_post(endpoint: str, payload: dict, api_key: Optional[str] = None) -> dict:
    """POST to an M2M endpoint and return the decoded JSON response."""
    key = api_key or _get_api_key()
    body = dict(payload)
    body["api_key"] = key

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{M2M_BASE}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def search_scenes(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
    sensors: Tuple[str, ...] = DEFAULT_SENSORS,
    api_key: Optional[str] = None,
) -> List[dict]:
    """
    Search Landsat scenes intersecting a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        (lon_min, lat_min, lon_max, lat_max) in WGS84 degrees.
    start_date, end_date : str
        Inclusive ISO dates, 'YYYY-MM-DD'.
    max_cloud : float
        Reject scenes with cloud cover above this percentage.  Landsat's
        ``meants`` field is the authoritative scene-level estimate.
    sensors : tuple of str
        M2M sensor codes, e.g. ('L8', 'L9').
    api_key : str, optional
        USGS M2M key.  Falls back to ``USGS_M2M_API_KEY``.

    Returns
    -------
    list of dict
        Raw M2M scene records, sorted by ascending cloud cover.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    payload = {
        "max_results": 200,
        "datasetCategory": "Real Land Data Constants",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft":  {"latitude": lat_min, "longitude": lon_min},
            "upperRight": {"latitude": lat_max, "longitude": lon_max},
        },
        "temporalFilter": {
            "startDate": start_date,
            "endDate": end_date,
        },
        "dataSetId": "landsat_ot_c2_l2",
    }

    resp = _m2m_post("scene-search", payload, api_key=api_key)
    if not resp.get("data"):
        print(f"  No scenes for {bbox} in {start_date}..{end_date}")
        return []

    wanted = tuple(s.upper() for s in sensors)
    scenes = [
        s for s in resp["data"]
        if s.get("displayId", "")[:2].upper() in wanted
        or s.get("sensorId", "").upper() in wanted
    ]

    # Filter by product ID: L2SP (surface reflectance) is preferred over L2SR
    # (raw surface reflectance, needs manual atmospheric correction).
    scenes = [
        s for s in scenes
        if any(s.get("displayId", "").startswith(p) for p in L2_PRODUCT_IDS)
    ]

    def cloud(s: dict) -> float:
        try:
            return float(s.get("cloudCover", 100.0))
        except (TypeError, ValueError):
            return 100.0

    scenes = [s for s in scenes if cloud(s) <= max_cloud]
    scenes.sort(key=cloud)
    return scenes


def _pc_sas_token() -> str:
    """Fetch (and cache) an anonymous SAS token for the PC Landsat collection."""
    global _PC_TOKEN
    if not _PC_TOKEN:
        with urllib.request.urlopen(PC_SAS_TOKEN, timeout=30) as resp:
            _PC_TOKEN = json.loads(resp.read())["token"]
    return _PC_TOKEN


def search_scenes_pc(
    bbox: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
    sensors: Tuple[str, ...] = DEFAULT_SENSORS,
    max_results: int = 200,
) -> List[dict]:
    """
    Search the keyless Planetary Computer mirror of Landsat C2 L2.

    Same parameters and record shape as :func:`search_scenes` (displayId,
    acquiredDate, cloudCover, WRS path/row), so pair selection, formatting
    and manifest writing are shared between backends.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    payload = {
        "collections": ["landsat-c2-l2"],
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": max_results,
    }
    req = urllib.request.Request(
        PC_STAC_SEARCH,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        items = json.loads(resp.read()).get("features", [])

    wanted = {s.upper() for s in sensors}
    scenes: List[dict] = []
    for item in items:
        props = item.get("properties", {})
        display_id = item.get("id", "")
        if not any(display_id.startswith(p) for p in L2_PRODUCT_IDS):
            continue

        # platform is 'landsat-8'/'landsat-9'; fall back to the LC08/LC09 id.
        platform = str(props.get("platform", ""))
        if "-" in platform:
            sat = "L" + platform.rsplit("-", 1)[-1]
        else:
            sat = "L" + display_id[2:4].lstrip("0")
        if sat.upper() not in wanted:
            continue

        cloud = float(props.get("eo:cloud_cover", 100.0))
        if cloud > max_cloud:
            continue

        scenes.append({
            "displayId":     display_id,
            "acquiredDate":  str(props.get("datetime", ""))[:10],
            "cloudCover":    cloud,
            "pathNumber":    props.get("landsat:wrs_path"),
            "rowNumber":     props.get("landsat:wrs_row"),
            "wrsPath":       props.get("landsat:wrs_path"),
            "wrsRow":        props.get("landsat:wrs_row"),
            "sensorId":      "OLI_TIRS" if display_id[2:3] == "C" else "TIRS_OLI",
            "satelliteId":   platform.upper(),
            "spacecraft_id": platform.upper(),
            "_backend":      "pc",
            "_assets": {
                k: a.get("href") for k, a in item.get("assets", {}).items()
            },
        })

    scenes.sort(key=lambda s: s["cloudCover"])
    return scenes


def format_scene(scene: dict) -> str:
    """Render a scene record as a one-line human-readable summary."""
    return (
        f"  {scene.get('displayId', '?'):<24} "
        f"{str(scene.get('acquiredDate', '?')):<12} "
        f"cloud={float(scene.get('cloudCover', 0)):>5.1f}%  "
        f"path={scene.get('pathNumber', '?'):<4} row={scene.get('rowNumber', '?'):<4} "
        f"wrs={scene.get('wrsPath', '?')}/{scene.get('wrsRow', '?')}"
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _scene_download_options(scene_id: str, api_key: Optional[str] = None) -> dict:
    """Fetch the per-scene product ID → download-options mapping from M2M."""
    resp = _m2m_post(
        "download-options",
        {
            "dataset": "landsat_ot_c2_l2",
            "displayId": scene_id,
        },
        api_key=api_key,
    )
    if resp.get("code") not in (None, 200, "200"):
        raise RuntimeError(
            f"download-options failed for {scene_id}: "
            f"{resp.get('code')} {resp.get('message')}"
        )
    return resp.get("data", [])


def _resolve_download_urls(
    scene_id: str,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return ``{product_id: https_url}`` for one scene.

    M2M returns relative URLs plus a DataSpace token; they are combined here
    into absolute HTTPS links.
    """
    urls: Dict[str, str] = {}

    for entry in _scene_download_options(scene_id, api_key=api_key):
        product_id = entry.get("id")
        if not product_id:
            continue

        # Only the surface-reflectance product and its QA band are needed;
        # thermal bands and thumbnails roughly triple the download size.
        is_sr = product_id.startswith(("LC08_L2SP", "LC09_L2SP",
                                       "LT08_L2SP", "LT09_L2SP"))
        is_qa = product_id.endswith("_QA_PIXEL")
        if not (is_sr or is_qa):
            continue

        for option in entry.get("downloadOptions", []):
            if option.get("available"):
                urls[product_id] = option["url"]

    return urls


def _download_file(url: str, dest: str) -> bool:
    """Stream a URL to disk atomically.  Returns True on success."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"

    req = urllib.request.Request(url, headers={"User-Agent": "LandslideAI/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
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
    except Exception as exc:                      # noqa: BLE001 - report and continue
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"    [warn] download failed: {exc}")
        return False


def download_scene(
    scene: dict,
    output_dir: str,
    api_key: Optional[str] = None,
    overwrite: bool = False,
) -> List[str]:
    """
    Download one Landsat C2 L2 scene's surface-reflectance bands.

    Parameters
    ----------
    scene : dict
        Scene record from :func:`search_scenes`.
    output_dir : str
        Directory to place the ``*.TIF`` band files in.
    api_key : str, optional
    overwrite : bool
        Re-download even if the band file already exists.

    Returns
    -------
    list of str
        Paths of the files actually written.
    """
    scene_id = scene.get("displayId", "unknown")
    print(f"\n  Downloading {scene_id} …")

    try:
        urls = _resolve_download_urls(scene_id, api_key=api_key)
    except Exception as exc:                      # noqa: BLE001
        print(f"    [error] could not resolve URLs: {exc}")
        return []

    if not urls:
        print("    [error] no surface-reflectance products available.")
        return []

    written: List[str] = []
    for product_id, url in sorted(urls.items()):
        fname = os.path.join(output_dir, f"{product_id}.TIF")
        if os.path.exists(fname) and not overwrite:
            print(f"    [cache] {os.path.basename(fname)}")
            written.append(fname)
            continue
        if _download_file(url, fname):
            written.append(fname)

    return written


def download_scene_pc(
    scene: dict,
    output_dir: str,
    overwrite: bool = False,
) -> List[str]:
    """
    Download one scene's surface-reflectance bands from the Planetary
    Computer mirror (no API key required).

    Writes ``{displayId}_{band}.TIF`` files whose suffixes match the
    ``preprocess_optical`` band discovery glob.

    Parameters
    ----------
    scene : dict
        Record produced by :func:`search_scenes_pc`.
    output_dir : str
        Directory to place the ``*.TIF`` band files in.
    overwrite : bool
        Re-download even if the band file already exists.

    Returns
    -------
    list of str
        Paths of the files actually present when finished.
    """
    display_id = scene.get("displayId", "unknown")
    print(f"\n  Downloading {display_id} (Planetary Computer) …")

    assets = scene.get("_assets") or {}
    written: List[str] = []
    for asset_key, band in PC_BAND_ASSETS.items():
        fname = os.path.join(output_dir, f"{display_id}_{band}.TIF")
        if os.path.exists(fname) and not overwrite:
            print(f"    [cache] {os.path.basename(fname)}")
            written.append(fname)
            continue
        href = assets.get(asset_key)
        if not href:
            print(f"    [warn] {display_id}: mirror has no '{asset_key}' asset")
            continue
        sep = "&" if "?" in href else "?"
        if _download_file(f"{href}{sep}{_pc_sas_token()}", fname):
            written.append(fname)

    if not written:
        print("    [error] no surface-reflectance bands retrieved.")

    return written


def _write_manifest(scene: dict, band_files: List[str], output_dir: str) -> str:
    """
    Write a sidecar JSON describing the scene.

    ``preprocess_optical.read_landsat_l2`` discovers bands by filename suffix,
    so the scene directory must contain the ``.TIF`` files directly -- the
    manifest is metadata only.
    """
    manifest = {
        "display_id":   scene.get("displayId"),
        "acquired":     scene.get("acquiredDate"),
        "cloud_cover":  scene.get("cloudCover"),
        "wrs_path":     scene.get("wrsPath"),
        "wrs_row":      scene.get("wrsRow"),
        "sensor_id":    scene.get("sensorId"),
        "satellite":    scene.get("satelliteId"),
        "spacecraft_id":scene.get("spacecraftId"),
        "files":        [os.path.basename(f) for f in band_files],
        "product_ids":  sorted(
            os.path.basename(f).rsplit("_", 4)[0] for f in band_files
        ),
    }
    path = os.path.join(output_dir, "scene_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def select_pair(
    pre_scenes: List[dict],
    post_scenes: List[dict],
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Pick the lowest-cloud pre-event and post-event scene from each list.

    Returns ``(pre, post)``; either may be ``None`` if its list is empty.
    """
    pre = min(pre_scenes, key=lambda s: float(s.get("cloudCover", 100))) if pre_scenes else None
    post = min(post_scenes, key=lambda s: float(s.get("cloudCover", 100))) if post_scenes else None
    return pre, post


def acquire_event(
    event_key: str,
    output_dir: str = "data/raw",
    max_cloud: float = 20.0,
    sensors: Tuple[str, ...] = DEFAULT_SENSORS,
    api_key: Optional[str] = None,
    download: bool = True,
    backend: Optional[str] = None,
) -> Dict[str, object]:
    """
    Find and (optionally) download the Landsat pre/post pair for one event.

    Parameters
    ----------
    event_key : str
        Key from ``INDIA_LANDSLIDE_EVENTS`` in download_sentinel.py.
    output_dir : str
        Root raw-data directory.  Landsat lands in ``{output_dir}/landsat/{event_key}/``.
    max_cloud : float
        Per-scene cloud cover ceiling, percent.
    sensors : tuple of str
        M2M sensor codes to include.
    api_key : str, optional
        USGS M2M key.  Falls back to ``USGS_M2M_API_KEY``.
    download : bool
        When False, only print the scene list.
    backend : {'auto', 'm2m', 'pc'}, optional
        Acquisition backend.  ``auto`` (default) picks M2M when
        ``USGS_M2M_API_KEY`` is set and the keyless Planetary Computer
        mirror otherwise.

    Returns
    -------
    dict
        ``{'event': …, 'pre': …, 'post': …, 'files': {...}}``
    """
    from download_sentinel import INDIA_LANDSLIDE_EVENTS

    if event_key not in INDIA_LANDSLIDE_EVENTS:
        raise ValueError(
            f"Unknown event '{event_key}'. "
            f"Available: {sorted(INDIA_LANDSLIDE_EVENTS)}"
        )

    ev = INDIA_LANDSLIDE_EVENTS[event_key]
    event_date = ev["date"]
    pre_start, post_end = _event_windows(event_date)
    backend = resolve_backend(backend)

    # Event date splits the window in two.
    print(f"\n{'=' * 78}")
    print(f"{ev['name']}   ({event_date})")
    print(f"  bbox  : {ev['bbox']}")
    print(f"  window: {pre_start} → {post_end}, cloud ≤ {max_cloud}%")
    print(f"  backend: {backend}"
          f"{' (keyless mirror)' if backend == 'pc' else ''}")
    print(f"{'=' * 78}")

    if backend == "m2m":
        all_scenes = search_scenes(
            ev["bbox"], pre_start, post_end,
            max_cloud=max_cloud, sensors=sensors, api_key=api_key,
        )
    else:
        all_scenes = search_scenes_pc(
            ev["bbox"], pre_start, post_end,
            max_cloud=max_cloud, sensors=sensors,
        )

    if not all_scenes:
        print("  No Landsat scenes met the cloud threshold.  Consider raising "
              "--max-cloud or widening the window.")
        return {"event": event_key, "pre": None, "post": None, "files": {}}

    print(f"\n  {len(all_scenes)} candidate scene(s):")
    for s in all_scenes:
        print(format_scene(s))

    pre_cand = [s for s in all_scenes if str(s.get("acquiredDate", "")) < event_date]
    post_cand = [s for s in all_scenes if str(s.get("acquiredDate", "")) > event_date]

    pre_scene, post_scene = select_pair(pre_cand, post_cand)

    print(f"\n  Selected PRE : {pre_scene.get('displayId') if pre_scene else 'NONE'}")
    print(f"  Selected POST: {post_scene.get('displayId') if post_scene else 'NONE'}")

    result: Dict[str, object] = {
        "event": event_key,
        "pre": pre_scene,
        "post": post_scene,
        "files": {},
    }

    if not download:
        print("\n  --search-only: skipping downloads.")
        return result

    files: Dict[str, List[str]] = {}
    for role, scene in (("pre", pre_scene), ("post", post_scene)):
        if scene is None:
            continue
        scene_dir = os.path.join(output_dir, "landsat", event_key, role)
        if backend == "m2m":
            written = download_scene(scene, scene_dir, api_key=api_key)
        else:
            written = download_scene_pc(scene, scene_dir)
        if written:
            _write_manifest(scene, written, scene_dir)
            files[role] = written
        result["files"][role] = scene_dir        # type: ignore[index]

    if not files:
        print("\n  Nothing downloaded.  See the scene list above for manual "
              "EarthExplorer retrieval.")
    else:
        print("\n  Landsat raw data ready.  Next:")
        for role, scene_dir in files.items():
            print(f"    python preprocess_optical.py --pre <{role}-scene-dir> "
                  f"--sensor landsat --output output/optical/{event_key}/{role}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_landsat_coverage(max_cloud: float = 20.0,
                           backend: Optional[str] = None) -> None:
    """Print the Landsat scene count and best cloud cover per catalogued event."""
    from download_sentinel import INDIA_LANDSLIDE_EVENTS

    backend = resolve_backend(backend)

    print("=" * 92)
    print(f"{'Event':<16} | {'Date':<11} | {'Scenes':>6} | {'Best cloud':>10} | Candidates")
    print("-" * 92)

    for key, ev in INDIA_LANDSLIDE_EVENTS.items():
        pre_start, post_end = _event_windows(ev["date"])
        try:
            if backend == "m2m":
                scenes = search_scenes(
                    ev["bbox"], pre_start, post_end, max_cloud=max_cloud
                )
            else:
                scenes = search_scenes_pc(
                    ev["bbox"], pre_start, post_end, max_cloud=max_cloud
                )
        except RuntimeError:
            print(f"{key:<16} | {ev['date']:<11} | {'—':>6} | {'no API key':>10} |")
            continue

        best = min((float(s.get("cloudCover", 100)) for s in scenes), default=None)
        names = ", ".join(s.get("displayId", "?") for s in scenes[:3])
        print(
            f"{key:<16} | {ev['date']:<11} | {len(scenes):>6} | "
            f"{(f'{best:.1f}%' if best is not None else '—'):>10} | {names}"
        )

    print("=" * 92)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Landsat Collection 2 Level-2 downloader for India landslide events.\n"
            "Replaces Sentinel-2 in the optical slot: 30 m reflectance, 16-day\n"
            "per-satellite revisit, USGS M2M distribution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--event", help="Event key from the India landslide catalog.")
    target.add_argument(
        "--all-events",
        action="store_true",
        help="Print scene availability for every catalogued event.",
    )
    target.add_argument(
        "--list-events",
        action="store_true",
        help="List catalogued events with their Landsat windows.",
    )

    p.add_argument(
        "--max-cloud", type=float, default=20.0,
        help="Per-scene cloud cover ceiling in percent (default: 20).",
    )
    p.add_argument(
        "--sensors", nargs="+", default=list(DEFAULT_SENSORS),
        help="M2M sensor codes to include (default: L8 L9).",
    )
    p.add_argument(
        "--output", default="data/raw",
        help="Raw data root (default: data/raw).",
    )
    p.add_argument(
        "--search-only", action="store_true",
        help="Print candidate scenes without downloading.",
    )
    p.add_argument(
        "--api-key", default=None,
        help="USGS M2M key (default: $USGS_M2M_API_KEY).",
    )
    p.add_argument(
        "--backend", choices=["auto", "m2m", "pc"], default="auto",
        help="Acquisition backend: 'auto' uses M2M when USGS_M2M_API_KEY is "
             "set, otherwise the keyless Planetary Computer mirror (default).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-download bands that already exist locally.",
    )

    args = p.parse_args()

    if args.list_events:
        from download_sentinel import INDIA_LANDSLIDE_EVENTS

        print("=" * 84)
        print(f"{'Event':<16} | {'Date':<11} | Landsat window")
        print("-" * 84)
        for key, ev in INDIA_LANDSLIDE_EVENTS.items():
            pre_start, post_end = _event_windows(ev["date"])
            print(f"{key:<16} | {ev['date']:<11} | {pre_start} → {post_end}")
        print("=" * 84)
        return

    if args.all_events:
        list_landsat_coverage(max_cloud=args.max_cloud, backend=args.backend)
        return

    time.sleep(0)   # keep rate limiting obvious to callers that wrap this
    try:
        acquire_event(
            event_key=args.event,
            output_dir=args.output,
            max_cloud=args.max_cloud,
            sensors=tuple(args.sensors),
            api_key=args.api_key,
            download=not args.search_only,
            backend=args.backend,
        )
    except RuntimeError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()