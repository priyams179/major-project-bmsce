"""Full-scene inference: apply model to each event, save mask GeoTIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from .dataset import DATA_DIR, load_pair, list_events, normalize_stack, valid_mask, NODATA
from .model import SiameseUNet

OUT_DIR = Path("results/masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def infer_event(
    event_id: str,
    model: torch.nn.Module,
    device: torch.device,
    tile: int = 128,
    overlap: int = 32,
    thr: float = 0.5,
    data_dir: Path = DATA_DIR,
    out_dir: Path = OUT_DIR,
) -> Path | None:
    pre, post, prof = load_pair(event_id, data_dir)
    h, w = pre.shape[1:]

    pre_n = normalize_stack(pre)
    post_n = normalize_stack(post)
    diff = pre_n - post_n
    full = np.concatenate([pre_n, post_n, diff], 0)
    valid = valid_mask(pre, post, NODATA)

    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)

    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    for y in ys:
        for x in xs:
            patch = full[:, y : y + tile, x : x + tile]
            vp = valid[y : y + tile, x : x + tile]
            ph, pw = patch.shape[1:]
            if ph < tile or pw < tile:
                pad = ((0, 0), (0, tile - ph), (0, tile - pw))
                patch = np.pad(patch, pad)
                vp = np.pad(vp, ((0, tile - vp.shape[0]), (0, tile - vp.shape[1])))
            t = torch.from_numpy(patch[None].astype(np.float32))
            t = t * torch.from_numpy(vp[None, None].astype(np.float32))
            logits = model(t.to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            acc[y : y + ph, x : x + pw] += prob[:ph, :pw]
            cnt[y : y + ph, x : x + pw] += 1.0

    cnt[cnt == 0] = 1
    prob_map = acc / cnt
    mask = (prob_map > thr).astype(np.uint8)
    mask[~valid] = 0

    out_path = out_dir / f"{event_id}_mask.tif"
    profile = prof.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    # Also save probability map for QC.
    prob_path = out_dir / f"{event_id}_prob.tif"
    profile.update(dtype="float32", nodata=-1.0)
    with rasterio.open(prob_path, "w", **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    return out_path


def infer_all(
    checkpoint: Path = Path("checkpoints/best.pt"),
    event_ids: list[str] | None = None,
    thr: float = 0.5,
    tile: int = 128,
    overlap: int = 32,
) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SiameseUNet(
        in_ch=ckpt.get("in_ch", 9), base=ckpt.get("base", 32)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch')}, F1 {ckpt.get('f1'):.3f})")

    if event_ids is None:
        event_ids = list_events()

    outputs = []
    for eid in event_ids:
        p = infer_event(eid, model, device, tile, overlap, thr)
        if p:
            outputs.append(p)
            print(f"  {eid} -> {p.name}")
    print(f"Inferred {len(outputs)} events -> {OUT_DIR}")
    return outputs
