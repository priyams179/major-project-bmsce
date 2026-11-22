"""Label masks and spatial train/val/test splits for each event.

Kodagu 2018
    The published polygon inventory (Achaiah et al., 2024; 217 polygons,
    CC BY 4.0) is rasterised onto the Sentinel-2 10 m grid. This is real,
    independently mapped ground truth.

Wayanad 2024
    No public vector outline of the Mundakkai-Chooralmala debris flow exists
    (the NRSC / Charter impact maps and journal figures are raster/PDF only),
    and the original LandSentry repo ships just a *point* placemark. We
    therefore build a **seeded change reference mask**: pixels with a strong
    NDVI loss and low post-event NDVI, kept only if they belong to the
    connected change region that touches the published placemark. This
    isolates the single debris-flow track and rejects unrelated change
    elsewhere in the scene. It is a semi-automatic reference, not
    field-validated ground truth; its derivation is index-based, so Wayanad
    scores measure agreement with that reference (see REPORT.md caveats).

Splits (``split.tif``: 0 ignore, 1 train, 2 val, 3 test)
    Kodagu: 512 px (5.12 km) blocks randomly assigned (seeded) so that the
    landslide pixels are ~65 / 15 / 20 % train / val / test.
    Wayanad: one debris-flow track, so the scene is cut perpendicular to the
    flow direction: the upper track (source + transport zone) is train, a
    middle strip is val, and the lower track (runout / deposition through
    Chooralmala) is test. Whole blocks of background go with each part.

Usage:  python -m landsentry labels [--event kodagu|wayanad|all]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform as warp_xy
from scipy import ndimage as ndi

from . import config as C
from .prepare import write_tif

# Wayanad reference-mask parameters (documented in REPORT.md).
DNDVI_MIN = 0.20        # NDVI loss pre -> post
NDVI_POST_MAX = 0.40    # post-event surface must be non-vegetated
BRIDGE_PX = 3           # dilation used only to link fragments of the same track
SEED_RADIUS_PX = 30     # the track must pass within 300 m of the placemark
MIN_OBJ_PX = 20         # drop specks (< 0.2 ha)


def _read(name):
    d = C.event_dir(name)
    with rasterio.open(d / "pre.tif") as s:
        pre = s.read(list(range(1, 5))).astype(np.float32) / 1e4
        prof = s.profile
    with rasterio.open(d / "post.tif") as s:
        post = s.read(list(range(1, 5))).astype(np.float32) / 1e4
    return pre, post, prof


def ndvi(a):
    n, r = a[C.NIR_IDX], a[C.RED_IDX]
    return (n - r) / np.maximum(n + r, 1e-6)


def kodagu_mask(prof):
    import geopandas as gpd
    inv = gpd.read_file(C.KODAGU_INVENTORY).to_crs(prof["crs"])
    mask = rasterize([(g, 1) for g in inv.geometry if g is not None and not g.is_empty],
                     out_shape=(prof["height"], prof["width"]),
                     transform=prof["transform"], fill=0, dtype="uint8")
    return mask, {"n_polygons": int(len(inv))}


def wayanad_mask(pre, post, prof):
    dn = ndvi(pre) - ndvi(post)
    cand = (dn > DNDVI_MIN) & (ndvi(post) < NDVI_POST_MAX)
    cand &= (pre > 0).all(0) & (post > 0).all(0)
    cand = ndi.binary_opening(cand, iterations=1)
    lab, n = ndi.label(ndi.binary_dilation(cand, iterations=BRIDGE_PX),
                       structure=np.ones((3, 3)))
    xs, ys = warp_xy("EPSG:4326", prof["crs"], [C.WAYANAD_SEED_LONLAT[0]],
                     [C.WAYANAD_SEED_LONLAT[1]])
    col, row = ~prof["transform"] * (xs[0], ys[0])
    r, c = int(row), int(col)
    rr, cc = np.ogrid[: lab.shape[0], : lab.shape[1]]
    near = (rr - r) ** 2 + (cc - c) ** 2 <= SEED_RADIUS_PX ** 2
    ids = np.unique(lab[near & (lab > 0)])
    if ids.size == 0:
        raise RuntimeError("no change region near the Wayanad seed point")
    track = np.isin(lab, ids) & cand
    track = ndi.binary_closing(track, iterations=1) & ndi.binary_dilation(cand, iterations=1)
    track = ndi.binary_fill_holes(track)
    lab2, n2 = ndi.label(track, structure=np.ones((3, 3)))
    sizes = ndi.sum(track, lab2, range(1, n2 + 1))
    track = np.isin(lab2, 1 + np.nonzero(sizes >= MIN_OBJ_PX)[0])
    info = {"seed_rowcol": [r, c], "dndvi_min": DNDVI_MIN, "ndvi_post_max": NDVI_POST_MAX,
            "bridge_px": BRIDGE_PX, "seed_radius_px": SEED_RADIUS_PX,
            "area_km2": float(track.sum() * 1e-4),
            "candidate_change_px_scene": int(cand.sum())}
    return track.astype(np.uint8), info


# --------------------------------------------------------------------------- #
def kodagu_split(mask, valid, block=512, seed=7, target=(0.65, 0.15, 0.20), tries=2000):
    H, W = mask.shape
    by, bx = int(np.ceil(H / block)), int(np.ceil(W / block))
    pos = np.zeros((by, bx))
    for i in range(by):
        for j in range(bx):
            pos[i, j] = mask[i * block:(i + 1) * block, j * block:(j + 1) * block].sum()
    rng = np.random.default_rng(seed)
    best, best_err = None, 1e9
    tot = pos.sum()
    for _ in range(tries):
        perm = rng.permutation(by * bx)
        n_val, n_test = round(0.15 * by * bx), round(0.20 * by * bx)
        lab = np.ones(by * bx, np.uint8)
        lab[perm[:n_val]] = 2
        lab[perm[n_val:n_val + n_test]] = 3
        share = np.array([pos.ravel()[lab == k].sum() / tot for k in (1, 2, 3)])
        err = np.abs(share - target).sum()
        if err < best_err:
            best, best_err = lab.reshape(by, bx), err
    split = np.kron(best, np.ones((block, block), np.uint8))[:H, :W]
    split[~valid] = 0
    return split.astype(np.uint8)


def make_wayanad_split(mask, valid, prof, gap=16):
    """Cut perpendicular to the flow axis (principal axis of the track)."""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    v = np.linalg.eigh(np.cov(np.stack([ys - cy, xs - cx])))[1][:, -1]
    rr, cc = np.mgrid[: mask.shape[0], : mask.shape[1]]
    proj = (rr - cy) * v[0] + (cc - cx) * v[1]
    # Orient the axis so the placemark (on the lower track, Chooralmala) is
    # at the + end: low projection = upper track / source zone.
    sx, sy = warp_xy("EPSG:4326", prof["crs"], [C.WAYANAD_SEED_LONLAT[0]],
                     [C.WAYANAD_SEED_LONLAT[1]])
    col, row = ~prof["transform"] * (sx[0], sy[0])
    if (row - cy) * v[0] + (col - cx) * v[1] < 0:
        proj = -proj
    pt = proj[mask.astype(bool)]
    q_tr, q_va = np.quantile(pt, [0.55, 0.70])   # 55 % train / 15 % val / 30 % test
    split = np.zeros(mask.shape, np.uint8)
    split[proj < q_tr - gap / 2] = 1
    split[(proj >= q_tr + gap / 2) & (proj < q_va - gap / 2)] = 2
    split[proj >= q_va + gap / 2] = 3
    split[~valid] = 0
    return split, {"cut_quantiles": [0.55, 0.70], "gap_px": gap}


def qc_figure(name, pre, post, mask, split, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import _rgb
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    a, b = _rgb(pre), _rgb(post)
    ax[0].imshow(a); ax[0].set_title(f"{name}: pre (S2 L1C median)")
    ax[1].imshow(b); ax[1].set_title("post")
    ax[2].imshow(b); ax[2].contour(mask, [0.5], colors="red", linewidths=0.6)
    ax[2].set_title(f"label mask ({mask.sum()*1e-4:.2f} km²)")
    cm = matplotlib.colors.ListedColormap(["#222222", "#4C9BE8", "#F2B134", "#E8574C"])
    ax[3].imshow(split, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    ax[3].contour(mask, [0.5], colors="white", linewidths=0.5)
    ax[3].set_title("split: blue train / amber val / red test")
    for x in ax:
        x.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def build(name: str):
    pre, post, prof = _read(name)
    valid = (pre > 0).all(0) & (post > 0).all(0)
    if name == "kodagu":
        mask, info = kodagu_mask(prof)
        split = kodagu_split(mask, valid)
        sinfo = {"scheme": "512px blocks, seeded random, target 65/15/20 by positives"}
    else:
        mask, info = wayanad_mask(pre, post, prof)
        split, sinfo = make_wayanad_split(mask, valid, prof)
        sinfo["scheme"] = "cut across flow axis: upper track train, middle val, lower track test"
    mask[~valid] = 0
    d = C.event_dir(name)
    write_tif(d / "mask.tif", mask, prof["transform"], prof["crs"])
    write_tif(d / "split.tif", split, prof["transform"], prof["crs"])
    stats = {k: {"px": int((split == v).sum()), "pos_px": int(mask[split == v].sum())}
             for k, v in (("train", 1), ("val", 2), ("test", 3))}
    meta = {"event": name, "label_source": C.EVENTS[name].label_source, **info,
            "mask_px": int(mask.sum()), "mask_frac": float(mask.mean()),
            "split": {**sinfo, **stats}}
    (d / "labels.json").write_text(json.dumps(meta, indent=2))
    qc_figure(name, pre, post, mask, split, d / "qc_labels.png")
    print(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="all", choices=["all", *C.EVENTS])
    a = ap.parse_args(argv)
    for n in (C.EVENTS if a.event == "all" else [a.event]):
        build(n)


if __name__ == "__main__":
    main()
