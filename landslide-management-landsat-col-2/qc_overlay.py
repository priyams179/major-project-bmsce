"""Visual QC overlays: PRE RGB | POST RGB | ensemble mask on POST.

Usage: python qc_overlay.py [event_id ...]
Default: wayanad_in_2024 acapulco_mx_2023 wajima_jp_2024
Output: qc/<event_id>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.dataset import DATA_DIR, NODATA, load_pair

MASK_DIR = Path("results/masks_ens")
OUT_DIR = Path("qc")
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_EVENTS = ["wayanad_in_2024", "acapulco_mx_2023", "wajima_jp_2024"]


def rgb_from(stack: np.ndarray) -> np.ndarray:
    """v2 stack [blue,green,red,nir,swir1,swir2,slope] -> natural-color RGB."""
    rgb = stack[[2, 1, 0]]  # red, green, blue
    out = np.stack([rgb[0], rgb[1], rgb[2]], -1)
    valid = np.all(stack[:6] != NODATA, axis=0)
    out = np.clip((out + 0.2) / 1.2, 0, 1)
    out[~valid] = 0.0
    return out.astype(np.float32)


def overlay_event(eid: str) -> Path | None:
    pre, post, _ = load_pair(eid, DATA_DIR)
    mask_path = MASK_DIR / f"{eid}_mask.tif"
    if not mask_path.exists():
        print(f"{eid}: no ensemble mask at {mask_path}")
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(bool)

    pre_rgb, post_rgb = rgb_from(pre), rgb_from(post)
    ov = post_rgb.copy()
    ov[mask] = ov[mask] * 0.35 + np.array([1.0, 0.1, 0.1]) * 0.65

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, img, title in zip(
        axes,
        [pre_rgb, post_rgb, ov],
        [f"{eid} PRE", f"{eid} POST", "ensemble mask (red)"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    frac = mask.mean()
    axes[2].set_title(f"ensemble mask (red) — {frac*100:.2f}% of scene", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"{eid}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    events = sys.argv[1:] or DEFAULT_EVENTS
    for eid in events:
        p = overlay_event(eid)
        if p:
            print(f"-> {p}")


if __name__ == "__main__":
    main()
