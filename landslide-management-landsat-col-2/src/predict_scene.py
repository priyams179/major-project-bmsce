"""Run the trained models on a full scene: per-tile landslide CLASSIFICATION
(PatchClassifier) + pixel SEGMENTATION (DeepUNet), stitched back to the scene
grid and vectorised to georeferenced polygons.

Outputs (in --out dir):
  seg_prob.tif        float32 landslide probability (scene CRS, 10 m)
  seg_mask.tif        uint8 binary mask at the segmentation threshold
  landslides.geojson  polygons in EPSG:4326 + area_km2 + mean prob
  tiles.csv           per 128x128 tile: row/col, lon/lat, cls prob + label
  overlay.png         RGB (B04,B03,B02) with the mask in red
  summary.json        scene-level counts / mapped area
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.warp import transform, transform_geom

from .model import DeepUNet, PatchClassifier, tta_logits, tta_logits_cls

CKPT_DIR = Path("checkpoints")
TILE = 128


def load_models(device, seg_ckpt="best_l4s.pt", cls_ckpt="best_l4s_cls.pt"):
    seg_s = torch.load(CKPT_DIR / seg_ckpt, map_location=device, weights_only=False)
    seg = DeepUNet(in_ch=14, base=seg_s.get("base", 64)).to(device).eval()
    seg.load_state_dict(seg_s["model"])
    cls_s = torch.load(CKPT_DIR / cls_ckpt, map_location=device, weights_only=False)
    clf = PatchClassifier(in_ch=14, base=cls_s.get("base", 64)).to(device).eval()
    clf.load_state_dict(cls_s["model"])
    # Frozen thresholds from the honest val sweeps (overridable).
    return seg, clf, {"seg_thr": 0.65, "cls_thr": 0.35}


def _normalize(stack, valid, mode, seg_ckpt_stats=None):
    """Per-band standardisation. mode='scene' uses this scene's own stats
    (robust to domain shift); mode='l4s' uses the model's train stats."""
    x = np.nan_to_num(stack).astype(np.float32)
    if mode == "l4s" and seg_ckpt_stats is not None:
        mean = np.asarray(seg_ckpt_stats["mean"], np.float32)[:, None, None]
        std = np.asarray(seg_ckpt_stats["std"], np.float32)[:, None, None]
    else:
        v = valid[None]
        n = max(valid.sum(), 1)
        mean = (x * v).reshape(14, -1).sum(1)[:, None, None] / n
        var = ((x - mean) ** 2 * v).reshape(14, -1).sum(1)[:, None, None] / n
        std = np.sqrt(np.maximum(var, 1e-12))
    return (x - mean) / np.maximum(std, 1e-6)


@torch.no_grad()
def predict_scene(stack, profile, valid, *, norm="scene", stride=TILE,
                  gate=False, tta=True, seg_thr=0.65, cls_thr=0.35,
                  min_area_px=16, device=None, seg_ckpt_stats=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg, clf, _ = load_models(device)
    H, W = profile["height"], profile["width"]
    xin = _normalize(stack, valid, norm, seg_ckpt_stats)

    prob_sum = np.zeros((H, W), np.float32)
    prob_cnt = np.zeros((H, W), np.float32)
    cls_gate = np.zeros((H, W), np.float32)
    tiles = []

    ys = list(range(0, max(H - TILE, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - TILE, 0) + 1, stride)) or [0]
    if ys[-1] != max(H - TILE, 0):
        ys.append(max(H - TILE, 0))
    if xs[-1] != max(W - TILE, 0):
        xs.append(max(W - TILE, 0))

    for yy in ys:
        for xx in xs:
            h = min(TILE, H - yy)
            w = min(TILE, W - xx)
            patch = np.zeros((14, TILE, TILE), np.float32)
            patch[:, :h, :w] = xin[:, yy:yy + h, xx:xx + w]
            vtile = valid[yy:yy + h, xx:xx + w]
            if vtile.mean() < 0.05:        # skip near-empty (edge/nodata) tiles
                continue
            t = torch.from_numpy(patch[None]).to(device)

            clp = float(torch.sigmoid(
                tta_logits_cls(clf, t) if tta else clf(t))[0].cpu())
            sp = torch.sigmoid(tta_logits(seg, t) if tta else seg(t))[0, 0].cpu().numpy()

            prob_sum[yy:yy + h, xx:xx + w] += sp[:h, :w]
            prob_cnt[yy:yy + h, xx:xx + w] += 1.0
            if clp >= cls_thr:
                cls_gate[yy:yy + h, xx:xx + w] = 1.0

            cx, cy = profile["transform"] * (xx + w / 2, yy + h / 2)
            lon, lat = transform(profile["crs"], "EPSG:4326", [cx], [cy])
            tiles.append({"row": yy, "col": xx, "lon": round(lon[0], 5),
                          "lat": round(lat[0], 5), "cls_prob": round(clp, 4),
                          "landslide": int(clp >= cls_thr)})

    prob = np.where(prob_cnt > 0, prob_sum / np.maximum(prob_cnt, 1), 0.0)
    prob *= valid
    mask = (prob > seg_thr) & (valid > 0)
    if gate:
        mask &= cls_gate > 0
    return {"prob": prob.astype(np.float32), "mask": mask.astype(np.uint8),
            "tiles": tiles, "profile": profile, "seg_thr": seg_thr,
            "cls_thr": cls_thr, "min_area_px": min_area_px}


# --------------------------------------------------------------------------- #
def _write_raster(path, arr, profile, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=profile["height"],
                       width=profile["width"], count=1, dtype=dtype,
                       crs=profile["crs"], transform=profile["transform"],
                       compress="deflate") as dst:
        dst.write(arr.astype(dtype), 1)


def _vectorize(mask, prob, profile, min_area_px):
    from shapely.geometry import shape
    px_area = abs(profile["transform"].a * profile["transform"].e)  # m^2
    feats = []
    for geom, val in shapes(mask, mask=mask.astype(bool),
                            transform=profile["transform"]):
        if val != 1:
            continue
        poly = shape(geom)
        if poly.area < min_area_px * px_area:
            continue
        g4326 = transform_geom(profile["crs"], "EPSG:4326", geom)
        pm = shape(g4326)
        feats.append({
            "type": "Feature", "geometry": g4326,
            "properties": {"area_km2": round(poly.area / 1e6, 4),
                           "centroid_lon": round(pm.centroid.x, 5),
                           "centroid_lat": round(pm.centroid.y, 5)}})
    return feats


def save_outputs(res, out_dir, stack=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof = res["profile"]
    _write_raster(out / "seg_prob.tif", res["prob"], prof, "float32")
    _write_raster(out / "seg_mask.tif", res["mask"], prof, "uint8")

    feats = _vectorize(res["mask"], res["prob"], prof, res["min_area_px"])
    with open(out / "landslides.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, indent=2)

    import csv
    with open(out / "tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "col", "lon", "lat",
                                          "cls_prob", "landslide"])
        w.writeheader()
        w.writerows(res["tiles"])

    n_ls_tiles = sum(t["landslide"] for t in res["tiles"])
    area = sum(fe["properties"]["area_km2"] for fe in feats)
    summary = {"n_tiles": len(res["tiles"]), "n_landslide_tiles": n_ls_tiles,
               "n_polygons": len(feats), "mapped_area_km2": round(area, 4),
               "seg_threshold": res["seg_thr"], "cls_threshold": res["cls_thr"]}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if stack is not None:
        _overlay(stack, res["mask"], res["tiles"], out / "overlay.png")
    print(f"  -> {out}/ : {len(feats)} polygons, {area:.3f} km², "
          f"{n_ls_tiles}/{len(res['tiles'])} tiles flagged")
    return summary


def _overlay(stack, mask, tiles, path):
    """RGB scene with pixel-level landslide mask (red) PLUS diagonal striped
    hatching over every tile the classifier flagged as landslide (the
    higher-accuracy tile-level signal)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    H, W = stack.shape[1], stack.shape[2]
    rgb = np.stack([stack[3], stack[2], stack[1]], -1)  # B04,B03,B02
    rgb = np.clip(rgb / (np.percentile(rgb[rgb > 0], 99) + 1e-6), 0, 1) if (rgb > 0).any() else rgb
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb)

    # Tile-level: diagonal stripes over classifier-flagged tiles (drawn first,
    # under the pixel mask so the exact scar stays clearly visible).
    n_flagged = 0
    for t in tiles or []:
        if not t.get("landslide"):
            continue
        n_flagged += 1
        w = min(TILE, W - t["col"])
        h = min(TILE, H - t["row"])
        ax.add_patch(Rectangle(
            (t["col"], t["row"]), w, h, facecolor=(1, 1, 0, 0.07),
            edgecolor=(1, 1, 0, 0.85), hatch="////", linewidth=1.1, zorder=3))

    # Pixel-level segmentation mask in red (on top).
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.65,
              zorder=4)

    ax.set_title("Landslide detection — red: segmented pixels | "
                 "yellow stripes: tiles classified landslide")
    ax.legend(handles=[
        Patch(facecolor="red", alpha=0.65, label="segmented landslide pixels"),
        Patch(facecolor=(1, 1, 0, 0.15), edgecolor=(1, 1, 0, 0.9), hatch="////",
              label=f"tile classified landslide ({n_flagged})")],
        loc="lower right", fontsize=8, framealpha=0.85)
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
