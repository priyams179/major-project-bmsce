"""Landsat-5 historical Indian landslide model: train on pre/post Landsat-5
(TM) scenes of 2007-2012 Indian events (downloaded cloud-filtered by
download_landsat5_india.py) and test on held-out events.

Honesty: these historical events have **point locations only** (NASA catalog),
no pixel masks. So training uses change-detection **pseudo-labels** (same
multi-index consensus as the main event pipeline), and the held-out "accuracy"
is a **catalog-proximity detection rate** (does the model flag landslide change
within R km of the catalogued point?) + visual QC -- not a true mask F1, which
is impossible without real masks.

Pipeline: `labels` -> pseudo masks; `train` -> best_landsat5.pt;
`test` -> held-out detection rate / proximity.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from .dataset import LandslidePairs, list_events, load_pair, split_events, \
    NODATA, extract_landsat_bands, normalize_stack
from .indices import multi_index_mask, valid_mask
from .model import SiameseUNet, FocalTverskyBCELoss, metrics, tta_logits

L5_DIR = Path("data/Landsat5_India")
L5_MASK = Path("data/pseudo_masks_l5")
CKPT = Path("checkpoints/best_landsat5.pt")
PATCH = 128
# Landsat-5 is cloud-starved over the monsoon Western Ghats (0% detection, §4.3c)
# so this model is scoped to Himalaya / Northeast; the Western Ghats is covered
# by the Sentinel-2 models (HR-GLDD Kodagu, L4S classifier) instead.
DROP_REGIONS = ("Western Ghats", "Sahyadri")


def _region(lat: float, lon: float) -> str:
    if 8 <= lat <= 16.5 and 73.5 <= lon <= 78:
        return "Western Ghats"
    if 16.5 < lat <= 21 and 72 <= lon <= 76:
        return "Sahyadri"
    if 22 <= lat <= 29 and 88 <= lon <= 97:
        return "Northeast"
    if 28 <= lat <= 35 and 74 <= lon <= 81:
        return "Himalaya"
    return "Other"


def _scope_events(events: list[str]) -> list[str]:
    """Keep only events outside the cloud-limited DROP_REGIONS (via manifest)."""
    man_path = L5_DIR / "manifest.csv"
    if not man_path.exists():
        return events
    import csv as _csv
    man = {r["id"]: r for r in _csv.DictReader(open(man_path))}
    kept = [e for e in events if e not in man
            or _region(float(man[e]["lat"]), float(man[e]["lon"])) not in DROP_REGIONS]
    dropped = len(events) - len(kept)
    if dropped:
        print(f"Scoped to Himalaya/NE: dropped {dropped} cloud-limited "
              f"{'/'.join(DROP_REGIONS)} events")
    return kept


# --------------------------------------------------------------------------- #
def make_labels() -> None:
    L5_MASK.mkdir(parents=True, exist_ok=True)
    events = list_events(L5_DIR)
    print(f"Pseudo-labels for {len(events)} Landsat-5 Indian events...")
    kept = 0
    for eid in events:
        pre, post, _ = load_pair(eid, L5_DIR)
        m = multi_index_mask(extract_landsat_bands(pre),
                             extract_landsat_bands(post), NODATA)
        np.save(L5_MASK / f"{eid}.npy", m)
        kept += int(m.sum() > 0)
        print(f"  {eid}: {m.mean()*100:.2f}% landslide px")
    print(f"Saved -> {L5_MASK} ({kept}/{len(events)} events have signal)")


# --------------------------------------------------------------------------- #
def _agg(ms):
    tp = sum(m["tp"] for m in ms); fp = sum(m["fp"] for m in ms)
    fn = sum(m["fn"] for m in ms); tn = sum(m["tn"] for m in ms)
    prec = tp / (tp + fp + 1e-6); rec = tp / (tp + fn + 1e-6)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec + 1e-6),
            "iou": tp / (tp + fp + fn + 1e-6)}


def train(epochs=40, batch_size=16, lr=3e-4, base=64, seed=42,
          num_workers=4, test_frac=0.2) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    events = list_events(L5_DIR)
    if not events:
        raise FileNotFoundError(f"No events in {L5_DIR} (run download_landsat5_india.py)")
    events = _scope_events(events)  # Himalaya/NE only (L5 can't do monsoon W.Ghats)
    tr_ev, te_ev = split_events(events, val_frac=test_frac, seed=seed)
    print(f"Landsat-5 India: {len(tr_ev)} train / {len(te_ev)} held-out events")
    json.dump({"train": tr_ev, "test": te_ev},
              open(L5_DIR / "split.json", "w"), indent=2)

    tr = LandslidePairs(tr_ev, PATCH, 64, augment=True, data_dir=L5_DIR,
                        mask_dir=L5_MASK)
    va = LandslidePairs(te_ev, PATCH, 64, augment=False, data_dir=L5_DIR,
                        mask_dir=L5_MASK)
    tr_dl = DataLoader(tr, batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=True, drop_last=True)
    va_dl = DataLoader(va, batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True)

    in_ch = int(tr[0]["x"].shape[0])
    model = SiameseUNet(in_ch=in_ch, base=base).to(device)
    print(f"SiameseUNet in_ch={in_ch} (3-band L5 pre|post|diff)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = FocalTverskyBCELoss(pos_weight=5.0, alpha=0.3, beta=0.7)

    best = {"f1": -1.0}
    CKPT.parent.mkdir(exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tm = []
        for b in tr_dl:
            x, y, v = b["x"].to(device), b["y"].to(device), b["valid"].to(device)
            if v.sum() < 100:
                continue
            opt.zero_grad()
            loss = crit(model(x), y, v)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tm.append(metrics(model(x).detach(), y, v))
        sched.step()
        model.eval(); vm = []
        with torch.no_grad():
            for b in va_dl:
                x, y, v = b["x"].to(device), b["y"].to(device), b["valid"].to(device)
                vm.append(metrics(model(x), y, v))
        vs = _agg(vm)
        print(f"Epoch {ep:02d}/{epochs} ({time.time()-t0:.0f}s) | "
              f"val(pseudo) F1 {vs['f1']:.3f} P {vs['precision']:.3f} "
              f"R {vs['recall']:.3f} IoU {vs['iou']:.3f}")
        if vs["f1"] > best["f1"]:
            best = dict(vs, epoch=ep)
            torch.save({"model": model.state_dict(), "in_ch": in_ch, "base": base,
                        "epoch": ep, "train_events": tr_ev, "test_events": te_ev,
                        "dataset": "landsat5_india"}, CKPT)
            print(f"  -> saved best (pseudo F1 {vs['f1']:.3f})")
    print(f"Best val(pseudo) F1 {best['f1']:.3f} -> {CKPT}")
    return best


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _infer_scene(model, pre, post, device, thr=0.5):
    """Full-scene landslide probability for a 3-band pre/post pair."""
    H, W = pre.shape[1:]
    pre_n, post_n = normalize_stack(pre), normalize_stack(post)
    diff = pre_n - post_n
    valid = valid_mask(pre, post, NODATA)
    x = np.concatenate([pre_n, post_n, diff], 0) * valid[None]
    prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - PATCH, 0) + 1, 64)) or [0]
    xs = list(range(0, max(W - PATCH, 0) + 1, 64)) or [0]
    for yy in ys:
        for xx in xs:
            patch = np.zeros((x.shape[0], PATCH, PATCH), np.float32)
            h = min(PATCH, H - yy); w = min(PATCH, W - xx)
            patch[:, :h, :w] = x[:, yy:yy + h, xx:xx + w]
            t = torch.from_numpy(patch[None]).to(device)
            p = torch.sigmoid(tta_logits(model, t))[0, 0].cpu().numpy()
            prob[yy:yy + h, xx:xx + w] += p[:h, :w]
            cnt[yy:yy + h, xx:xx + w] += 1
    prob = np.where(cnt > 0, prob / np.maximum(cnt, 1), 0) * valid
    return prob


def test(thr=0.5, radius_km=3.0, min_area_px=12) -> dict:
    """Held-out accuracy: detection rate within radius_km of the catalog point."""
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from shapely.geometry import shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(CKPT, map_location=device, weights_only=False)
    model = SiameseUNet(in_ch=st["in_ch"], base=st["base"]).to(device).eval()
    model.load_state_dict(st["model"])
    man = {r["id"]: r for r in csv.DictReader(open(L5_DIR / "manifest.csv"))}
    test_events = st["test_events"]

    rows, hits, dists = [], 0, []
    for eid in test_events:
        if eid not in man:
            continue
        pre, post, prof = load_pair(eid, L5_DIR)
        prob = _infer_scene(model, pre, post, device, thr)
        mask = (prob > thr).astype(np.uint8)
        lat0, lon0 = float(man[eid]["lat"]), float(man[eid]["lon"])
        # nearest detected polygon centroid to the catalogued point
        best_d = None
        with rasterio.open(L5_DIR / f"{eid}_PRE.tif") as src:
            tfm, crs = src.transform, src.crs
        for geom, val in shapes(mask, mask=mask.astype(bool), transform=tfm):
            if val != 1:
                continue
            poly = shape(geom)
            if poly.area < min_area_px * abs(tfm.a * tfm.e):
                continue
            g = shape(transform_geom(crs, "EPSG:4326", geom))
            d = _haversine(lat0, lon0, g.centroid.y, g.centroid.x)
            best_d = d if best_d is None else min(best_d, d)
        det = best_d is not None and best_d <= radius_km
        hits += int(det)
        if best_d is not None:
            dists.append(best_d)
        rows.append({"event": eid, "lat": lat0, "lon": lon0,
                     "detected": int(det),
                     "nearest_km": round(best_d, 2) if best_d is not None else None})
        print(f"  {eid}: {'HIT' if det else 'miss'} "
              f"(nearest {best_d:.2f} km)" if best_d is not None
              else f"  {eid}: miss (no detection)")

    n = len(rows)
    rate = hits / max(n, 1)
    result = {"n_test_events": n, "radius_km": radius_km,
              "detection_rate": round(rate, 3), "n_detected": hits,
              "median_nearest_km": round(float(np.median(dists)), 2) if dists else None,
              "per_event": rows}
    with open("checkpoints/landsat5_test.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n=== Held-out Landsat-5 India accuracy ===")
    print(f"  detection rate within {radius_km} km: {rate*100:.1f}% "
          f"({hits}/{n} events)")
    if dists:
        print(f"  median nearest-detection distance: {np.median(dists):.2f} km")
    print("  -> checkpoints/landsat5_test.json")
    return result


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
