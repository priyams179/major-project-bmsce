"""Dataset: load pre/post Landsat pairs, patch extraction, augmentation."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .indices import multi_index_mask, valid_mask

DATA_DIR = Path("data/Landslide_Landsat_Global_v2")
# v1 (3-band): [nir, swir1, swir2]
# v2 (7-band): [blue, green, red, nir, swir1, swir2, slope] (+valid_obs skipped)
NODATA = -9999.0


def _band_indices(n_bands: int) -> dict:
    """Locate nir/swir1/swir2 (+slope if present) from band count."""
    if n_bands >= 7:
        return {"nir": 3, "swir1": 4, "swir2": 5, "slope": 6, "refl": 6}
    return {"nir": 0, "swir1": 1, "swir2": 2, "slope": None, "refl": 3}


def list_events(data_dir: Path = DATA_DIR) -> list[str]:
    """Event IDs that have both PRE and POST tifs."""
    events = []
    for f in sorted(data_dir.glob("*_PRE.tif")):
        eid = f.name[: -len("_PRE.tif")]
        if (data_dir / f"{eid}_POST.tif").exists():
            events.append(eid)
    return events


def load_stack(path: Path) -> tuple[np.ndarray, dict]:
    """Read GeoTIFF -> float32 [bands,H,W] (reflectance + optional slope)."""
    with rasterio.open(path) as src:
        arr = src.read()  # [C,H,W]
        profile = src.profile
    bi = _band_indices(arr.shape[0])
    keep = list(range(bi["refl"]))
    if bi["slope"] is not None:
        keep.append(bi["slope"])
    stack = arr[keep].astype(np.float32)
    return stack, profile


def load_pair(event_id: str, data_dir: Path = DATA_DIR):
    """Return (pre[refl(+slope),H,W], post[same], profile)."""
    pre, prof = load_stack(data_dir / f"{event_id}_PRE.tif")
    post, _ = load_stack(data_dir / f"{event_id}_POST.tif")
    return pre, post, prof


def normalize_stack(stack: np.ndarray) -> np.ndarray:
    """Reflectance roughly [-0.2, 1.0] -> clip and scale to [0,1]. NaN -> 0.
    Slope (if present as last band) scaled by /90 (degrees)."""
    out = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=-0.2)
    if out.shape[0] >= 7:  # v2: last band is slope in degrees
        refl = np.clip(out[:6], -0.2, 1.0)
        refl = (refl + 0.2) / 1.2
        slope = np.clip(out[6:7], 0.0, 90.0) / 90.0
        return np.concatenate([refl, slope], 0).astype(np.float32)
    out = np.clip(out, -0.2, 1.0)
    out = (out - (-0.2)) / 1.2
    return out.astype(np.float32)


def extract_landsat_bands(stack: np.ndarray) -> np.ndarray:
    """Select nir, swir1, swir2 as [3,H,W] for spectral indices."""
    bi = _band_indices(stack.shape[0])
    return stack[[bi["nir"], bi["swir1"], bi["swir2"]]]


class LandslidePairs(Dataset):
    """Patch dataset over all events.

    Each item: dict with
      pre   [6,H,W]  normalized pre (nir,swir1,swir2) + dNBR-like extras
      post  [6,H,W]  normalized post + extras
      x     [9,W,W]  model input: pre3 | post3 | (pre-post)3
      y     [1,H,W]  binary mask
      valid [1,H,W]  valid-pixel mask
    """

    def __init__(
        self,
        event_ids: list[str],
        patch_size: int = 128,
        stride: int = 64,
        augment: bool = False,
        data_dir: Path = DATA_DIR,
        use_pseudo: bool = True,
        mask_dir: Path | None = Path("data/pseudo_masks_v2"),
    ):
        self.events = event_ids
        self.ps = patch_size
        self.stride = stride
        self.augment = augment
        self.data_dir = data_dir
        self.use_pseudo = use_pseudo
        self.mask_dir = mask_dir

        self.samples: list[tuple[str, int, int]] = []
        self._cache: dict[str, tuple] = {}

        for eid in self.events:
            pre, post, _ = load_pair(eid, data_dir)
            h, w = pre.shape[1:]
            for y in range(0, max(h - patch_size, 0) + 1, stride):
                for x in range(0, max(w - patch_size, 0) + 1, stride):
                    self.samples.append((eid, y, x))
            if h <= patch_size and w <= patch_size:
                self.samples.append((eid, 0, 0))

        print(f"Dataset: {len(self.events)} events, {len(self.samples)} patches")

    def __len__(self):
        return len(self.samples)

    def _get_event(self, eid: str):
        if eid not in self._cache:
            pre, post, prof = load_pair(eid, self.data_dir)
            self._cache[eid] = (pre, post, prof)
        return self._cache[eid]

    def _get_mask(self, eid: str, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        if self.use_pseudo and self.mask_dir is not None:
            mp = self.mask_dir / f"{eid}.npy"
            if mp.exists():
                return np.load(mp).astype(np.uint8)
        return multi_index_mask(
            extract_landsat_bands(pre), extract_landsat_bands(post), NODATA
        )

    def __getitem__(self, idx: int):
        eid, y, x = self.samples[idx]
        pre, post, _ = self._get_event(eid)
        ps = self.ps

        pre_p = pre[:, y : y + ps, x : x + ps]
        post_p = post[:, y : y + ps, x : x + ps]

        # Pad if scene smaller than patch.
        ph = ps - pre_p.shape[1]
        pw = ps - pre_p.shape[2]
        if ph > 0 or pw > 0:
            pad = ((0, 0), (0, max(ph, 0)), (0, max(pw, 0)))
            pre_p = np.pad(pre_p, pad, constant_values=NODATA)
            post_p = np.pad(post_p, pad, constant_values=NODATA)

        mask_full = self._get_mask(eid, pre, post)
        y_p = mask_full[y : y + ps, x : x + ps]
        if y_p.shape != (ps, ps):
            y_p = np.pad(y_p, ((0, ps - y_p.shape[0]), (0, ps - y_p.shape[1])))

        valid = valid_mask(pre_p, post_p, NODATA)

        # Feature stack.
        pre_n = normalize_stack(pre_p)
        post_n = normalize_stack(post_p)
        diff = pre_n - post_n

        x_in = np.concatenate([pre_n, post_n, diff], axis=0)  # [9,H,W]
        y_in = y_p[None].astype(np.float32)
        v_in = valid[None].astype(np.float32)

        # Zero invalid pixels in input.
        x_in *= v_in

        if self.augment:
            x_in, y_in, v_in = self._augment(x_in, y_in, v_in)

        return {
            "x": torch.from_numpy(x_in.copy()),
            "y": torch.from_numpy(y_in.copy()),
            "valid": torch.from_numpy(v_in.copy()),
            "event": eid,
        }

    @staticmethod
    def _augment(x, y, v):
        # Horizontal / vertical flip, 90-degree rotations.
        if np.random.rand() < 0.5:
            x, y, v = x[:, :, ::-1], y[:, :, ::-1], v[:, :, ::-1]
        if np.random.rand() < 0.5:
            x, y, v = x[:, ::-1, :], y[:, ::-1, :], v[:, ::-1, :]
        k = np.random.randint(4)
        if k:
            x = np.rot90(x, k, axes=(1, 2))
            y = np.rot90(y, k, axes=(1, 2))
            v = np.rot90(v, k, axes=(1, 2))
        # Brightness jitter on first 6 channels.
        if np.random.rand() < 0.5:
            s = 1.0 + np.random.uniform(-0.1, 0.1)
            x[:6] = np.clip(x[:6] * s, 0, 1)
        return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(v)


def split_events(event_ids: list[str], val_frac: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(event_ids))
    n_val = max(1, int(len(event_ids) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [event_ids[i] for i in range(len(event_ids)) if i not in val_idx]
    val = [event_ids[i] for i in range(len(event_ids)) if i in val_idx]
    return train, val
