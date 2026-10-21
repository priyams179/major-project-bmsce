"""Spectral indices and pseudo-label generation.

NBR = (NIR - SWIR2) / (NIR + SWIR2)
dNBR = NBR_pre - NBR_post   (positive => vegetation loss / bare soil exposed)

Pseudo-labels via Otsu threshold on valid dNBR pixels + morphological cleanup.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import disk
from skimage.morphology.binary import binary_closing, binary_opening


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio. Inputs float reflectance."""
    denom = nir + swir2
    out = np.full_like(nir, np.nan, dtype=np.float32)
    valid = denom > 1e-6
    out[valid] = (nir[valid] - swir2[valid]) / denom[valid]
    return out


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index."""
    denom = nir + swir1
    out = np.full_like(nir, np.nan, dtype=np.float32)
    valid = denom > 1e-6
    out[valid] = (nir[valid] - swir1[valid]) / denom[valid]
    return out


def dnbr(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """dNBR from pre and post 3-band stacks (nir, swir1, swir2) each [3,H,W]."""
    pre_nbr = nbr(pre[0], pre[2])
    post_nbr = nbr(post[0], post[2])
    return pre_nbr - post_nbr


def dndmi(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """dNDMI: moisture drop after event -> positive."""
    return ndmi(pre[0], pre[1]) - ndmi(post[0], post[1])


def dswir2(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Relative SWIR2 increase (bare soil exposure)."""
    return (post[2] - pre[2]) / (np.abs(pre[2]) + 0.05)


def valid_mask(pre: np.ndarray, post: np.ndarray, nodata: float = -9999.0) -> np.ndarray:
    """Pixels valid (not nodata, finite) in both images, all bands."""
    ok_pre = np.all(pre != nodata, axis=0) & np.all(np.isfinite(pre), axis=0)
    ok_post = np.all(post != nodata, axis=0) & np.all(np.isfinite(post), axis=0)
    return ok_pre & ok_post


def _top_frac_mask(d: np.ndarray, valid: np.ndarray, target_frac: float,
                   abs_floor: float, max_frac: float) -> np.ndarray:
    """Conservative positive-change mask from a difference index."""
    vals = d[valid]
    if np.nanstd(vals) < 1e-6:
        return np.zeros(d.shape, dtype=bool)
    thr = max(abs_floor, float(np.percentile(vals, (1 - target_frac) * 100)))
    cand = (d > thr) & valid
    for _ in range(8):
        if cand.sum() / max(valid.sum(), 1) <= max_frac:
            break
        thr *= 1.25
        cand = (d > thr) & valid
    return cand


def multi_index_mask(
    pre_stack: np.ndarray,
    post_stack: np.ndarray,
    nodata: float = -9999.0,
    min_area_px: int = 25,
    votes_required: int = 2,
    target_frac: float = 0.04,
) -> np.ndarray:
    """Consensus pseudo-label from dNBR + dNDMI + dSWIR2.

    A pixel is landslide if >= votes_required indices agree on
    strong positive change. Much higher precision than single dNBR.
    """
    valid = valid_mask(pre_stack, post_stack, nodata)
    mask = np.zeros(pre_stack.shape[1:], dtype=np.uint8)
    if valid.sum() < 100:
        return mask

    m1 = _top_frac_mask(dnbr(pre_stack, post_stack), valid, target_frac, 0.20, 0.10)
    m2 = _top_frac_mask(dndmi(pre_stack, post_stack), valid, target_frac, 0.15, 0.10)
    m3 = _top_frac_mask(dswir2(pre_stack, post_stack), valid, target_frac, 0.30, 0.12)

    votes = m1.astype(np.uint8) + m2.astype(np.uint8) + m3.astype(np.uint8)
    cand = (votes >= votes_required) & valid

    cand = binary_opening(cand, disk(2))
    cand = binary_closing(cand, disk(3))

    try:
        from skimage.measure import label, regionprops

        lab = label(cand)
        keep = np.zeros_like(cand)
        for rp in regionprops(lab):
            if rp.area >= min_area_px:
                keep[lab == rp.label] = True
        cand = keep
    except Exception:
        pass

    mask[cand] = 1
    return mask


def pseudo_mask(
    pre_stack: np.ndarray,
    post_stack: np.ndarray,
    nodata: float = -9999.0,
    min_area_px: int = 25,
    target_frac: float = 0.05,
    abs_floor: float = 0.20,
    max_frac: float = 0.12,
) -> np.ndarray:
    """Binary pseudo-label mask [H,W] uint8 {0,1} from dNBR.

    Conservative strategy (Otsu alone over-labels seasonal change):
      thr = max(abs_floor, (1-target_frac) percentile of dNBR)
      If resulting fraction > max_frac, raise percentile until under cap.
    Landslides: positive dNBR (NBR dropped after event).
    """
    valid = valid_mask(pre_stack, post_stack, nodata)
    d = dnbr(pre_stack, post_stack)

    mask = np.zeros(d.shape, dtype=np.uint8)
    if valid.sum() < 100:
        return mask

    vals = d[valid]
    if np.nanstd(vals) < 1e-6:
        return mask

    # Base: top target_frac of change, but at least abs_floor.
    thr = max(abs_floor, float(np.percentile(vals, (1 - target_frac) * 100)))

    cand = (d > thr) & valid

    # Cap coverage: iteratively raise threshold.
    for _ in range(8):
        frac = cand.sum() / max(valid.sum(), 1)
        if frac <= max_frac:
            break
        thr *= 1.25
        cand = (d > thr) & valid

    # Morphological cleanup.
    cand = binary_opening(cand, disk(2))
    cand = binary_closing(cand, disk(3))

    try:
        from skimage.measure import label, regionprops

        lab = label(cand)
        keep = np.zeros_like(cand)
        for rp in regionprops(lab):
            if rp.area >= min_area_px:
                keep[lab == rp.label] = True
        cand = keep
    except Exception:
        pass

    mask[cand] = 1
    return mask
