"""Scene loading, normalisation, spatial splits and the training sampler.

Each event is stored as full-scene rasters (``pre.tif``, ``post.tif``,
``mask.tif``, ``split.tif``). Training samples random crops on the fly from
the *train* region only (a crop must lie entirely inside it, so no imagery
from val/test leaks into training context); validation and testing run
sliding-window inference over the whole scene and score only the pixels of the
requested split.

Normalisation is *per scene*: one set of per-band 2nd/98th percentiles is
computed over the union of the pre and post composites and applied to both
dates. Using the same statistics for both dates preserves real reflectance
change; computing them per scene removes event-to-event radiometric offsets
(different haze, sun angle, year) -- the cross-event failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import minimum_filter
from torch.utils.data import Dataset

from . import config as C

SPLITS = {"train": 1, "val": 2, "test": 3}


def scene_norm_stats(pre: np.ndarray, post: np.ndarray, valid: np.ndarray):
    """Per-band (lo, hi) = 2nd / 98th percentiles over pre U post valid pixels."""
    lo, hi = [], []
    for b in range(pre.shape[0]):
        v = np.concatenate([pre[b][valid], post[b][valid]])
        if v.size > 2_000_000:
            v = np.random.default_rng(0).choice(v, 2_000_000, replace=False)
        p2, p98 = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
        lo.append(p2)
        hi.append(max(p98, p2 + 1e-3))
    return np.array(lo, np.float32), np.array(hi, np.float32)


def normalise(pre, post, valid=None, stats=None):
    """Return the stacked model input ``[2C, H, W]`` float32 and the stats."""
    if valid is None:
        valid = (pre != 0).all(0) & (post != 0).all(0)
    if stats is None:
        stats = scene_norm_stats(pre, post, valid)
    lo, hi = stats
    f = lambda a: np.clip((a - lo[:, None, None]) / (hi - lo)[:, None, None], -0.5, 1.5)
    x = np.concatenate([f(pre), f(post)], 0).astype(np.float32)
    x[:, ~valid] = 0.0
    return x, stats


@dataclass
class Scene:
    name: str
    x: np.ndarray          # [8, H, W] normalised pre|post
    mask: np.ndarray       # [H, W] uint8 0/1
    valid: np.ndarray      # [H, W] bool (both dates have data)
    split: np.ndarray      # [H, W] uint8 (0 ignore, 1 train, 2 val, 3 test)
    transform: object
    crs: object
    rgb_pre: np.ndarray    # [H, W, 3] for figures
    rgb_post: np.ndarray
    dndvi: np.ndarray      # [H, W] NDVI(pre) - NDVI(post), for the index baseline

    def region(self, split: str | None) -> np.ndarray:
        r = self.valid.copy()
        if split not in (None, "all"):
            r &= self.split == SPLITS[split]
        return r


def _rgb(a: np.ndarray) -> np.ndarray:
    """True-colour display image; per-band 2-98 % stretch removes TOA haze cast."""
    rgb = np.stack([a[i] for i in C.RGB_IDX], -1).astype(np.float32)
    ok = (rgb > 0).all(-1)
    for k in range(3):
        v = rgb[..., k][ok]
        lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
        rgb[..., k] = (rgb[..., k] - lo) / (hi - lo + 1e-6)
    return np.clip(rgb, 0, 1) ** 0.9


def load_scene(name: str, data_dir: Path | None = None) -> Scene:
    d = Path(data_dir or C.event_dir(name))
    with rasterio.open(d / "pre.tif") as s:
        pre = s.read(list(range(1, 5))).astype(np.float32) / 1e4
        tfm, crs = s.transform, s.crs
    with rasterio.open(d / "post.tif") as s:
        post = s.read(list(range(1, 5))).astype(np.float32) / 1e4
    with rasterio.open(d / "mask.tif") as s:
        mask = s.read(1).astype(np.uint8)
    with rasterio.open(d / "split.tif") as s:
        split = s.read(1).astype(np.uint8)
    valid = (pre > 0).all(0) & (post > 0).all(0)
    x, _ = normalise(pre, post, valid)

    def ndvi(a):
        return (a[C.NIR_IDX] - a[C.RED_IDX]) / np.maximum(a[C.NIR_IDX] + a[C.RED_IDX], 1e-6)

    dndvi = (ndvi(pre) - ndvi(post)).astype(np.float32)
    return Scene(name, x, mask, valid, split, tfm, crs, _rgb(pre), _rgb(post), dndvi)


class CropSampler(Dataset):
    """Random crops from the train region of one or more scenes.

    ``pos_frac`` of the crops are centred near a random landslide pixel (the
    classes are ~0.3 % positive); the rest are uniform over the region.
    Augmentation: D4 flips/rotations + independent per-date radiometric
    jitter (gain/offset), which mimics the inter-event radiometric shifts the
    model must be robust to for cross-event transfer.
    """

    def __init__(self, scenes: list[Scene], crop: int = 128, length: int = 3200,
                 pos_frac: float = 0.5, split: str = "train", augment: bool = True,
                 seed: int = 0):
        self.scenes, self.crop, self.length = scenes, crop, length
        self.pos_frac, self.augment = pos_frac, augment
        self.rng = np.random.default_rng(seed)
        self.centres, self.pos_centres = [], []
        h = crop // 2
        for s in scenes:
            region = s.region(split)
            # top-left corners whose crop lies fully inside the region
            ok = minimum_filter(region.astype(np.uint8), size=crop, mode="constant", cval=0) > 0
            # minimum_filter is centred; shift to top-left convention
            tl = np.zeros_like(ok)
            tl[: ok.shape[0] - h, : ok.shape[1] - h] = ok[h:, h:]
            ys, xs = np.nonzero(tl)
            self.centres.append((ys, xs))
            # crops that contain a landslide pixel near their centre
            my, mx = np.nonzero(s.mask.astype(bool) & region)
            py = np.clip(my - h + self.rng.integers(-h // 2, h // 2 + 1, my.size), 0, None)
            px = np.clip(mx - h + self.rng.integers(-h // 2, h // 2 + 1, mx.size), 0, None)
            keep = (py < tl.shape[0]) & (px < tl.shape[1])
            keep[keep] = tl[py[keep], px[keep]]
            self.pos_centres.append((py[keep], px[keep]))
        w = np.array([c[0].size for c in self.centres], float)
        if w.sum() == 0:
            raise ValueError("no valid crop positions in the requested region")
        self.scene_p = w / w.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = self.rng
        k = rng.choice(len(self.scenes), p=self.scene_p)
        s = self.scenes[k]
        ys, xs = self.centres[k]
        pys, pxs = self.pos_centres[k]
        if pys.size and rng.random() < self.pos_frac:
            i = rng.integers(pys.size)
            y, x = pys[i], pxs[i]
        else:
            i = rng.integers(ys.size)
            y, x = ys[i], xs[i]
        c = self.crop
        img = s.x[:, y:y + c, x:x + c].copy()
        msk = s.mask[y:y + c, x:x + c].astype(np.float32)
        val = s.valid[y:y + c, x:x + c].astype(np.float32)
        if self.augment:
            r = rng.integers(4)
            img, msk, val = (np.rot90(a, r, axes=(-2, -1)) for a in (img, msk, val))
            if rng.random() < 0.5:
                img, msk, val = (a[..., ::-1] for a in (img, msk, val))
            nb = img.shape[0] // 2
            for t in range(2):              # independent jitter per date
                g = rng.normal(1.0, 0.08, (nb, 1, 1)).astype(np.float32)
                o = rng.normal(0.0, 0.04, (nb, 1, 1)).astype(np.float32)
                img[t * nb:(t + 1) * nb] = img[t * nb:(t + 1) * nb] * g + o
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(msk))[None],
                torch.from_numpy(np.ascontiguousarray(val))[None])
