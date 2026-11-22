"""Google Earth Engine access: cloud-masked Sentinel-2 composites downloaded
on a fixed UTM 10 m grid, so pre and post rasters are pixel-aligned by
construction (no resampling or co-registration step is needed afterwards).

Used by the offline data preparation (``prepare.py``).
"""

from __future__ import annotations

import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import transform_bounds

from . import config as C

_EE = None


def init(project: str | None = None):
    """Initialise Earth Engine once per process and return the ``ee`` module."""
    global _EE
    if _EE is None:
        import ee
        ee.Initialize(project=project or C.GEE_PROJECT)
        _EE = ee
    return _EE


def utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def s2_composite(region, start: str, end: str):
    """Median Sentinel-2 L1C composite with s2cloudless + QA60 cloud masking.

    Returns an int16 image with bands ``BANDS`` (TOA reflectance x 1e4) plus
    ``n_obs`` (clear observations per pixel). Masked pixels are left masked so
    the caller can ``unmask(0)``.
    """
    ee = init()
    s2 = (ee.ImageCollection(C.S2_COLLECTION)
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.SCENE_CLOUD_MAX)))
    prob = ee.ImageCollection(C.S2_CLOUD_PROB).filterBounds(region).filterDate(start, end)
    joined = ee.ImageCollection(ee.Join.saveFirst("cloud").apply(
        primary=s2, secondary=prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index")))

    def mask(img):
        img = ee.Image(img)
        cp = ee.Image(img.get("cloud")).select("probability")
        qa = img.select("QA60")
        clear = (cp.lt(C.CLOUD_PROB_MAX)
                 .And(qa.bitwiseAnd(1 << 10).eq(0))
                 .And(qa.bitwiseAnd(1 << 11).eq(0)))
        return img.select(list(C.BANDS)).updateMask(clear)

    col = joined.map(mask)
    med = col.median().select(list(C.BANDS))
    n_obs = col.select(C.BANDS[0]).count().rename("n_obs")
    return med.addBands(n_obs).toInt16()


def utm_grid(bbox, epsg: int, scale: int = C.SCALE_M):
    """Snap a lon/lat bbox to a UTM pixel grid -> (x0, y0, width, height)."""
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
    x0 = math.floor(xmin / scale) * scale
    y0 = math.ceil(ymax / scale) * scale
    w = int(math.ceil((xmax - x0) / scale))
    h = int(math.ceil((y0 - ymin) / scale))
    return x0, y0, w, h


def _fetch(img, bands, epsg, x0, y0, w, h, scale, retries=4):
    params = {"bands": list(bands), "crs": f"EPSG:{epsg}",
              "crs_transform": [scale, 0, x0, 0, -scale, y0],
              "dimensions": f"{w}x{h}", "format": "GEO_TIFF"}
    for attempt in range(retries):
        try:
            url = img.getDownloadURL(params)
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            with MemoryFile(data) as mf, mf.open() as ds:
                arr = ds.read()
            if arr.shape[1:] != (h, w):           # defensive: pad / crop
                out = np.zeros((arr.shape[0], h, w), arr.dtype)
                hh, ww = min(h, arr.shape[1]), min(w, arr.shape[2])
                out[:, :hh, :ww] = arr[:, :hh, :ww]
                arr = out
            return arr
        except Exception as e:  # network / EE quota hiccups
            if attempt == retries - 1:
                raise RuntimeError(f"GEE download failed: {e}") from e
            time.sleep(2 * (attempt + 1))


def download(img, bands, bbox, epsg: int | None = None, scale: int = C.SCALE_M,
             tile: int = 1024, workers: int = 6, progress=None):
    """Download ``img`` over ``bbox`` as a [C, H, W] array on a UTM grid.

    Returns ``(array, transform, crs_string)``. Large areas are split into
    ``tile`` x ``tile`` requests (EE caps a single download at ~48 MB).
    """
    ee = init()
    if epsg is None:
        epsg = utm_epsg_for((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    x0, y0, W, H = utm_grid(bbox, epsg, scale)
    img = ee.Image(img).unmask(0)
    out = np.zeros((len(bands), H, W), np.int16)
    jobs = [(r, c) for r in range(0, H, tile) for c in range(0, W, tile)]

    def run(rc):
        r, c = rc
        h, w = min(tile, H - r), min(tile, W - c)
        arr = _fetch(img, bands, epsg, x0 + c * scale, y0 - r * scale, w, h, scale)
        out[:, r:r + h, c:c + w] = arr
        return rc

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(run, jobs), 1):
            if progress:
                progress(i, len(jobs))
    transform = Affine(scale, 0, x0, 0, -scale, y0)
    return out, transform, f"EPSG:{epsg}"


def fetch_pre_post(bbox, pre: tuple[str, str], post: tuple[str, str],
                   epsg: int | None = None, progress=None):
    """Convenience: pre & post composites over ``bbox`` on the same grid.

    Returns ``(pre[4,H,W], post[4,H,W], n_obs_pre[H,W], n_obs_post[H,W],
    transform, crs)`` with reflectance as float32 in [0, ~1].
    """
    ee = init()
    region = ee.Geometry.Rectangle(list(bbox))
    bands = list(C.BANDS) + ["n_obs"]
    res = []
    for k, (d0, d1) in enumerate((pre, post)):
        cb = (lambda i, n, k=k: progress(k, i, n)) if progress else None
        arr, tfm, crs = download(s2_composite(region, d0, d1), bands, bbox,
                                 epsg=epsg, progress=cb)
        res.append(arr)
    pre_a, post_a = res
    return (pre_a[:4].astype(np.float32) / 1e4, post_a[:4].astype(np.float32) / 1e4,
            pre_a[4], post_a[4], tfm, crs)
