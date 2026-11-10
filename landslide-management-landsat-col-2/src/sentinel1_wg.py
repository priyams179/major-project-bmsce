"""Western Ghats landslide model on **Sentinel-1 C-band SAR** (cloud-penetrating).

Landsat-5 scored 0% on the monsoon Western Ghats (clouds). SAR sees through
clouds, so this model covers the events Landsat-5 could not. Input per scene:
VV, VH backscatter (dB) + slope (NASADEM); the model is bi-temporal on the two
polarisations: pre(VV,VH) | post(VV,VH) | Δ(VV,VH) = 6 channels (ΔVV, ΔVH are
the log-ratio change layers the literature finds most discriminative).

Labels: SAR change-detection **pseudo-labels** (significant |ΔVV|/|ΔVH| on
slopes; no pixel masks exist), so the held-out metric is the same
**catalog-proximity detection rate** as the Landsat-5 model, not a mask F1.

Pipeline: `labels` -> pseudo masks; `train` -> best_s1_wg.pt; `test`.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from skimage.morphology import binary_closing, binary_opening, disk
from torch.utils.data import DataLoader, Dataset

from .landsat5 import _haversine
from .model import (FocalTverskyBCELoss, PatchClassifier, SiameseUNet, metrics,
                    tta_logits, tta_logits_cls)

S1_DIR = Path("data/Sentinel1_WG")
S1_MASK = Path("data/pseudo_masks_s1_wg")
CKPT = Path("checkpoints/best_s1_wg.pt")
PATCH = 128


def load_pair(eid: str):
    """(pre[3,H,W], post[3,H,W], profile) with bands VV, VH, slope (dB / deg)."""
    with rasterio.open(S1_DIR / f"{eid}_PRE.tif") as s:
        pre = s.read().astype(np.float32); prof = s.profile
    with rasterio.open(S1_DIR / f"{eid}_POST.tif") as s:
        post = s.read().astype(np.float32)
    return pre, post, prof


def _norm(stack: np.ndarray) -> np.ndarray:
    """Fixed dB->[0,1] so pre/post share a scale and Δ = normalised log-ratio.
    VV ~ [-25,15], VH ~ [-30,0], slope/90."""
    out = np.nan_to_num(stack, nan=0.0).astype(np.float32)
    vv = np.clip((out[0] + 25) / 40.0, 0, 1)
    vh = np.clip((out[1] + 30) / 30.0, 0, 1)
    slope = np.clip(out[2], 0, 90) / 90.0
    return np.stack([vv, vh, slope], 0)


def _valid(pre, post):
    return np.all(np.isfinite(pre[:2]), 0) & np.all(np.isfinite(post[:2]), 0) \
        & np.any(pre[:2] != 0, 0) & np.any(post[:2] != 0, 0)


def sar_pseudo_mask(pre, post, min_area=15, slope_min=8.0):
    """Landslide pseudo-label: strong VV/VH backscatter change on slopes."""
    from scipy.ndimage import median_filter
    valid = _valid(pre, post)
    slope = np.nan_to_num(pre[2])
    # Speckle reduction: median-filter the log-ratio change before thresholding
    # so labels follow real scars, not single-pixel SAR noise.
    dvv = median_filter(np.nan_to_num(post[0] - pre[0]), size=3)
    dvh = median_filter(np.nan_to_num(post[1] - pre[1]), size=3)
    mask = np.zeros(pre.shape[1:], np.uint8)
    if valid.sum() < 100:
        return mask
    on_slope = (slope > slope_min) & valid

    def strong(d):
        v = np.abs(d)[on_slope]
        if v.size < 50 or np.std(v) < 1e-6:
            return np.zeros_like(d, bool)
        thr = max(2.0, float(np.percentile(v, 96)))  # >=2 dB change, top ~4%
        return (np.abs(d) > thr) & on_slope

    cand = strong(dvv) & strong(dvh)          # both polarisations agree
    cand = binary_opening(cand, disk(1))
    cand = binary_closing(cand, disk(2))
    try:
        from skimage.measure import label, regionprops
        lab = label(cand); keep = np.zeros_like(cand)
        for rp in regionprops(lab):
            if rp.area >= min_area:
                keep[lab == rp.label] = True
        cand = keep
    except Exception:
        pass
    mask[cand] = 1
    return mask


def list_events():
    return sorted(p.name[:-len("_PRE.tif")] for p in S1_DIR.glob("*_PRE.tif")
                  if (S1_DIR / f"{p.name[:-len('_PRE.tif')]}_POST.tif").exists())


def make_labels():
    S1_MASK.mkdir(parents=True, exist_ok=True)
    ev = list_events()
    print(f"SAR pseudo-labels for {len(ev)} Western Ghats events...")
    kept = 0
    for eid in ev:
        pre, post, _ = load_pair(eid)
        m = sar_pseudo_mask(pre, post)
        np.save(S1_MASK / f"{eid}.npy", m)
        kept += int(m.sum() > 0)
        print(f"  {eid}: {m.mean()*100:.2f}% landslide px")
    print(f"Saved -> {S1_MASK} ({kept}/{len(ev)} have signal)")


class S1Patches(Dataset):
    def __init__(self, events, augment=False):
        self.samples, self.cache = [], {}
        self.augment = augment
        for eid in events:
            pre, _, _ = load_pair(eid)
            H, W = pre.shape[1:]
            for y in range(0, max(H - PATCH, 0) + 1, 64):
                for x in range(0, max(W - PATCH, 0) + 1, 64):
                    self.samples.append((eid, y, x))
        print(f"S1 dataset: {len(events)} events, {len(self.samples)} patches")

    def __len__(self):
        return len(self.samples)

    def _get(self, eid):
        if eid not in self.cache:
            pre, post, _ = load_pair(eid)
            m = np.load(S1_MASK / f"{eid}.npy") if (S1_MASK / f"{eid}.npy").exists() \
                else sar_pseudo_mask(pre, post)
            self.cache[eid] = (pre, post, m)
        return self.cache[eid]

    def __getitem__(self, i):
        eid, y, x = self.samples[i]
        pre, post, m = self._get(eid)
        ps = PATCH
        pre_p, post_p = pre[:, y:y+ps, x:x+ps], post[:, y:y+ps, x:x+ps]
        mp = m[y:y+ps, x:x+ps]
        ph, pw = ps - pre_p.shape[1], ps - pre_p.shape[2]
        if ph > 0 or pw > 0:
            pad = ((0, 0), (0, ph), (0, pw))
            pre_p = np.pad(pre_p, pad); post_p = np.pad(post_p, pad)
            mp = np.pad(mp, ((0, ph), (0, pw)))
        v = _valid(pre_p, post_p).astype(np.float32)
        pn, qn = _norm(pre_p)[:2], _norm(post_p)[:2]  # VV,VH only to the model
        xin = np.concatenate([pn, qn, pn - qn], 0) * v[None]  # 6 ch
        if self.augment:
            if np.random.rand() < 0.5:
                xin, mp, v = xin[:, :, ::-1], mp[:, ::-1], v[:, ::-1]
            if np.random.rand() < 0.5:
                xin, mp, v = xin[:, ::-1], mp[::-1], v[::-1]
        return {"x": torch.from_numpy(np.ascontiguousarray(xin)),
                "y": torch.from_numpy(np.ascontiguousarray(mp[None].astype(np.float32))),
                "valid": torch.from_numpy(np.ascontiguousarray(v[None]))}


def _agg(ms):
    tp = sum(m["tp"] for m in ms); fp = sum(m["fp"] for m in ms)
    fn = sum(m["fn"] for m in ms)
    p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6)
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r+1e-6),
            "iou": tp/(tp+fp+fn+1e-6)}


def train(epochs=40, batch_size=16, lr=3e-4, base=64, seed=42, test_frac=0.25,
          num_workers=4):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    events = list_events()
    if not events:
        raise FileNotFoundError(f"No events in {S1_DIR} (run download_sentinel1_wg.py)")
    rng = np.random.RandomState(seed); idx = rng.permutation(len(events))
    nv = max(1, int(len(events) * test_frac))
    te = [events[i] for i in idx[:nv]]; tr = [events[i] for i in idx[nv:]]
    print(f"Sentinel-1 Western Ghats: {len(tr)} train / {len(te)} held-out events")

    tr_dl = DataLoader(S1Patches(tr, True), batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=True)
    va_dl = DataLoader(S1Patches(te, False), batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
    model = SiameseUNet(in_ch=6, base=base).to(device)
    print("SiameseUNet in_ch=6 (SAR pre[VV,VH]|post|diff)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = FocalTverskyBCELoss(pos_weight=5.0, alpha=0.3, beta=0.7)
    best = {"f1": -1.0}; CKPT.parent.mkdir(exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tm = []
        for b in tr_dl:
            x, y, v = b["x"].to(device), b["y"].to(device), b["valid"].to(device)
            if v.sum() < 100:
                continue
            opt.zero_grad(); loss = crit(model(x), y, v)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tm.append(metrics(model(x).detach(), y, v))
        sched.step()
        model.eval(); vm = []
        with torch.no_grad():
            for b in va_dl:
                x, y, v = b["x"].to(device), b["y"].to(device), b["valid"].to(device)
                vm.append(metrics(model(x), y, v))
        vs = _agg(vm)
        print(f"Epoch {ep:02d}/{epochs} ({time.time()-t0:.0f}s) | val(pseudo) "
              f"F1 {vs['f1']:.3f} P {vs['precision']:.3f} R {vs['recall']:.3f}")
        if vs["f1"] > best["f1"]:
            best = dict(vs, epoch=ep)
            torch.save({"model": model.state_dict(), "in_ch": 6, "base": base,
                        "epoch": ep, "train_events": tr, "test_events": te,
                        "dataset": "sentinel1_wg"}, CKPT)
            print(f"  -> saved best (pseudo F1 {vs['f1']:.3f})")
    print(f"Best val(pseudo) F1 {best['f1']:.3f} -> {CKPT}")
    return best


@torch.no_grad()
def _infer(model, pre, post, device):
    H, W = pre.shape[1:]
    pn, qn = _norm(pre)[:2], _norm(post)[:2]
    v = _valid(pre, post)
    x = np.concatenate([pn, qn, pn - qn], 0) * v[None]
    prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    for yy in range(0, max(H-PATCH, 0)+1, 64) or [0]:
        for xx in range(0, max(W-PATCH, 0)+1, 64) or [0]:
            p = np.zeros((6, PATCH, PATCH), np.float32)
            h, w = min(PATCH, H-yy), min(PATCH, W-xx)
            p[:, :h, :w] = x[:, yy:yy+h, xx:xx+w]
            t = torch.from_numpy(p[None]).to(device)
            pr = torch.sigmoid(tta_logits(model, t))[0, 0].cpu().numpy()
            prob[yy:yy+h, xx:xx+w] += pr[:h, :w]; cnt[yy:yy+h, xx:xx+w] += 1
    return np.where(cnt > 0, prob/np.maximum(cnt, 1), 0) * v


def test(thr=0.5, radius_km=3.0, min_area_px=12):
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from shapely.geometry import shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(CKPT, map_location=device, weights_only=False)
    model = SiameseUNet(in_ch=6, base=st["base"]).to(device).eval()
    model.load_state_dict(st["model"])
    man = {r["id"]: r for r in csv.DictReader(open(S1_DIR / "manifest.csv"))}
    rows, hits, dists = [], 0, []
    for eid in st["test_events"]:
        if eid not in man:
            continue
        pre, post, _ = load_pair(eid)
        prob = _infer(model, pre, post, device)
        mask = (prob > thr).astype(np.uint8)
        lat0, lon0 = float(man[eid]["lat"]), float(man[eid]["lon"])
        with rasterio.open(S1_DIR / f"{eid}_PRE.tif") as src:
            tfm, crs = src.transform, src.crs
        best_d = None
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
        rows.append({"event": eid, "detected": int(det),
                     "nearest_km": round(best_d, 2) if best_d is not None else None})
        print(f"  {eid}: {'HIT' if det else 'miss'}"
              + (f" (nearest {best_d:.2f} km)" if best_d is not None else " (no detection)"))
    n = len(rows); rate = hits / max(n, 1)
    res = {"sensor": "Sentinel-1 SAR", "region": "Western Ghats",
           "n_test_events": n, "radius_km": radius_km,
           "detection_rate": round(rate, 3), "n_detected": hits,
           "median_nearest_km": round(float(np.median(dists)), 2) if dists else None,
           "per_event": rows}
    json.dump(res, open("checkpoints/s1_wg_test.json", "w"), indent=2)
    print(f"\n=== Sentinel-1 Western Ghats held-out accuracy ===")
    print(f"  detection within {radius_km} km: {rate*100:.1f}% ({hits}/{n})")
    if dists:
        print(f"  median nearest-detection: {np.median(dists):.2f} km")
    print("  -> checkpoints/s1_wg_test.json")
    return res


# --------------------------------------------------------------------------- #
# REAL ground truth (Kodagu 2018 inventory, 217 polygons)
# --------------------------------------------------------------------------- #
GT_DIR = Path("data/Kodagu_SAR_GT")
CKPT_REAL = Path("checkpoints/best_s1_wg_real.pt")


def _load_gt():
    with rasterio.open(GT_DIR / "PRE.tif") as s:
        pre = s.read().astype(np.float32)
    with rasterio.open(GT_DIR / "POST.tif") as s:
        post = s.read().astype(np.float32)
    with rasterio.open(GT_DIR / "mask.tif") as s:
        mask = s.read(1).astype(np.uint8)
    return pre, post, mask


@torch.no_grad()
def _infer_stride(model, pre, post, device, stride=128, tta=False):
    H, W = pre.shape[1:]
    pn, qn = _norm(pre)[:2], _norm(post)[:2]
    v = _valid(pre, post)
    x = np.concatenate([pn, qn, pn - qn], 0) * v[None]
    prob = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    for yy in range(0, max(H-PATCH, 0)+1, stride):
        for xx in range(0, max(W-PATCH, 0)+1, stride):
            p = np.zeros((6, PATCH, PATCH), np.float32)
            h, w = min(PATCH, H-yy), min(PATCH, W-xx)
            p[:, :h, :w] = x[:, yy:yy+h, xx:xx+w]
            t = torch.from_numpy(p[None]).to(device)
            pr = torch.sigmoid(tta_logits(model, t) if tta else model(t))[0, 0].cpu().numpy()
            prob[yy:yy+h, xx:xx+w] += pr[:h, :w]; cnt[yy:yy+h, xx:xx+w] += 1
    return np.where(cnt > 0, prob/np.maximum(cnt, 1), 0) * v, v


def _px_metrics(prob, mask, valid, thr):
    pred = (prob > thr) & (valid > 0); gt = (mask > 0) & (valid > 0)
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
    return {"thr": round(thr, 2), "f1": 2*p*r/(p+r+1e-9), "precision": p,
            "recall": r, "iou": tp/(tp+fp+fn+1e-9)}


def eval_real(ckpt=CKPT, tag="pseudo-trained"):
    """Score a model against the REAL Kodagu inventory (pixel F1/IoU + sweep)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model = SiameseUNet(in_ch=6, base=st["base"]).to(device).eval()
    model.load_state_dict(st["model"])
    pre, post, mask = _load_gt()
    prob, valid = _infer_stride(model, pre, post, device, stride=128)
    sweep = [_px_metrics(prob, mask, valid, t) for t in np.arange(0.2, 0.81, 0.05)]
    best = max(sweep, key=lambda s: s["f1"])
    print(f"\n=== REAL Kodagu-2018 GT eval ({tag}, {int((mask>0).sum())} gt px) ===")
    print(f"  best F1 {best['f1']:.3f} @thr {best['thr']} | IoU {best['iou']:.3f} "
          f"P {best['precision']:.3f} R {best['recall']:.3f}")
    json.dump({"model": tag, "best": best, "sweep": sweep},
              open(f"checkpoints/s1_real_eval_{tag.split('-')[0]}.json", "w"), indent=2)
    return best


class _GTTiles(Dataset):
    """128-tiles from the real Kodagu scene; spatial block split by column."""
    def __init__(self, pre, post, mask, cols, augment=False):
        self.pre, self.post, self.mask = pre, post, mask
        self.augment = augment
        H, W = pre.shape[1:]
        self.samples = [(y, x) for y in range(0, H-PATCH+1, 64)
                        for x in cols if x + PATCH <= W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        y, x = self.samples[i]; ps = PATCH
        pre_p, post_p = self.pre[:, y:y+ps, x:x+ps], self.post[:, y:y+ps, x:x+ps]
        m = self.mask[y:y+ps, x:x+ps]
        v = _valid(pre_p, post_p).astype(np.float32)
        pn, qn = _norm(pre_p)[:2], _norm(post_p)[:2]
        xin = np.concatenate([pn, qn, pn-qn], 0) * v[None]
        if self.augment and np.random.rand() < 0.5:
            xin, m, v = xin[:, :, ::-1], m[:, ::-1], v[:, ::-1]
        return {"x": torch.from_numpy(np.ascontiguousarray(xin)),
                "y": torch.from_numpy(np.ascontiguousarray(m[None].astype(np.float32))),
                "valid": torch.from_numpy(np.ascontiguousarray(v[None]))}


def train_real(epochs=30, batch_size=16, lr=1e-4, seed=42):
    """Fine-tune the SAR model on REAL Kodagu masks; spatial (left/right) split
    to avoid leakage; report real pixel F1/IoU on the held-out right block."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre, post, mask = _load_gt()
    W = pre.shape[2]; cut = int(W * 0.7)
    tr_cols = list(range(0, cut-PATCH+1, 64))
    te_cols = list(range(cut, W-PATCH+1, 64))
    print(f"Real-GT fine-tune: scene {pre.shape[1]}x{W}, "
          f"train cols<{cut} / test cols>={cut}")
    tr = DataLoader(_GTTiles(pre, post, mask, tr_cols, True), batch_size,
                    shuffle=True, num_workers=4, drop_last=True)
    te_ds = _GTTiles(pre, post, mask, te_cols, False)

    st = torch.load(CKPT, map_location=device, weights_only=False)  # warm-start
    model = SiameseUNet(in_ch=6, base=st["base"]).to(device)
    model.load_state_dict(st["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = FocalTverskyBCELoss(pos_weight=10.0, alpha=0.25, beta=0.75)

    # held-out test block for real metrics (fixed grid over the right columns)
    def eval_test():
        model.eval()
        H = pre.shape[1]
        sub_pre = pre[:, :, te_cols[0]:]; sub_post = post[:, :, te_cols[0]:]
        sub_mask = mask[:, te_cols[0]:]
        prob, valid = _infer_stride(model, sub_pre, sub_post, device, 128)
        return max((_px_metrics(prob, sub_mask, valid, t)
                    for t in np.arange(0.2, 0.81, 0.05)), key=lambda s: s["f1"])

    best = {"f1": -1}
    for ep in range(1, epochs+1):
        model.train()
        for b in tr:
            x, y, v = b["x"].to(device), b["y"].to(device), b["valid"].to(device)
            if v.sum() < 50:
                continue
            opt.zero_grad(); loss = crit(model(x), y, v)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        m = eval_test()
        print(f"Epoch {ep:02d}/{epochs} | held-out REAL F1 {m['f1']:.3f} "
              f"IoU {m['iou']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} @thr {m['thr']}")
        if m["f1"] > best["f1"]:
            best = dict(m, epoch=ep)
            torch.save({"model": model.state_dict(), "base": st["base"], "in_ch": 6,
                        "real_f1": m["f1"], "dataset": "kodagu_real"}, CKPT_REAL)
    print(f"\nBest held-out REAL F1 {best['f1']:.3f} IoU {best['iou']:.3f} "
          f"(epoch {best['epoch']}) -> {CKPT_REAL}")
    json.dump(best, open("checkpoints/s1_real_finetune.json", "w"), indent=2)
    return best


# --------------------------------------------------------------------------- #
# Tile-level landslide PRESENCE classifier on real Kodagu GT (the 90%+ task)
# --------------------------------------------------------------------------- #
CKPT_CLS = Path("checkpoints/best_s1_wg_cls.pt")


def _cls_stack(pre_p, post_p):
    """7-ch tile: pre(VV,VH) | post(VV,VH) | Δ(VV,VH) | slope."""
    pn, qn = _norm(pre_p), _norm(post_p)
    return np.concatenate([pn[:2], qn[:2], pn[:2] - qn[:2], pn[2:3]], 0)


def _build_cls_tiles(pre, post, mask, cols, pos_px, augment=False):
    H = pre.shape[1]
    X, Y = [], []
    for y in range(0, H - PATCH + 1, 64):
        for x in cols:
            if x + PATCH > pre.shape[2]:
                continue
            pp, qp = pre[:, y:y+PATCH, x:x+PATCH], post[:, y:y+PATCH, x:x+PATCH]
            if _valid(pp, qp).mean() < 0.5:
                continue
            X.append(_cls_stack(pp, qp).astype(np.float32))
            Y.append(float(mask[y:y+PATCH, x:x+PATCH].sum() >= pos_px))
    return np.array(X), np.array(Y, np.float32)


def _cls_scores(P, G, thr):
    pred = P > thr; pos = G > 0.5
    tp = float((pred & pos).sum()); fp = float((pred & ~pos).sum())
    fn = float((~pred & pos).sum()); tn = float((~pred & ~pos).sum())
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
    tnr = tn/(tn+fp+1e-9)
    return {"thr": round(float(thr), 2), "f1": 2*p*r/(p+r+1e-9),
            "accuracy": (tp+tn)/(tp+tn+fp+fn+1e-9),
            "balanced_acc": 0.5*(r+tnr), "precision": p, "recall": r}


def _auc(P, G):
    pos, neg = P[G > 0.5], P[G <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order)+1)
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) /
                 (len(pos)*len(neg)))


def train_cls_real(epochs=60, batch_size=32, lr=3e-4, base=48, seed=42,
                   pos_px=20):
    """Tile-level landslide-presence classifier on real Kodagu masks.
    Spatial left/right split (no leakage). Reports F1 / balanced-accuracy / AUC
    (not raw accuracy alone — the tiles are imbalanced)."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pre, post, mask = _load_gt()
    W = pre.shape[2]; cut = int(W * 0.7)
    trX, trY = _build_cls_tiles(pre, post, mask, list(range(0, cut-PATCH+1, 64)), pos_px)
    teX, teY = _build_cls_tiles(pre, post, mask, list(range(cut, W-PATCH+1, 64)), pos_px)
    print(f"Tile classifier: {len(trX)} train ({trY.mean()*100:.0f}% pos) / "
          f"{len(teX)} test ({teY.mean()*100:.0f}% pos)")

    trX_t = torch.tensor(trX); trY_t = torch.tensor(trY)
    teX_t = torch.tensor(teX).to(device)
    model = PatchClassifier(in_ch=7, base=base).to(device)
    pos_w = torch.tensor([(trY == 0).sum()/max((trY == 1).sum(), 1)], device=device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(trX_t); best = {"f1": -1}
    for ep in range(1, epochs+1):
        model.train()
        for idx in torch.randperm(n).split(batch_size):
            x = trX_t[idx].to(device); y = trY_t[idx].to(device)
            # light flip augmentation
            if np.random.rand() < 0.5:
                x = torch.flip(x, dims=(3,))
            opt.zero_grad(); loss = crit(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            P = torch.sigmoid(tta_logits_cls(model, teX_t)).cpu().numpy()
        sweep = [_cls_scores(P, teY, t) for t in np.arange(0.3, 0.71, 0.05)]
        m = max(sweep, key=lambda s: s["f1"])
        if ep % 10 == 0 or m["f1"] > best["f1"]:
            print(f"Epoch {ep:02d}/{epochs} | held-out F1 {m['f1']:.3f} "
                  f"acc {m['accuracy']:.3f} bal-acc {m['balanced_acc']:.3f} "
                  f"P {m['precision']:.3f} R {m['recall']:.3f}")
        if m["f1"] > best["f1"]:
            best = dict(m, epoch=ep, auc=_auc(P, teY))
            torch.save({"model": model.state_dict(), "base": base, "in_ch": 7,
                        "dataset": "kodagu_cls"}, CKPT_CLS)
    print(f"\n=== Western Ghats SAR tile classifier (real GT, held-out) ===")
    print(f"  F1 {best['f1']*100:.1f}% | accuracy {best['accuracy']*100:.1f}% | "
          f"balanced-acc {best['balanced_acc']*100:.1f}% | AUC {best['auc']:.3f} "
          f"(thr {best['thr']}, epoch {best['epoch']})")
    json.dump(best, open("checkpoints/s1_cls_real.json", "w"), indent=2)
    return best
