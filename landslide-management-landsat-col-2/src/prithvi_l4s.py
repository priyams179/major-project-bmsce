"""Fine-tune the Prithvi-EO-1.0 geospatial foundation model (IBM-NASA) for
Landslide4Sense segmentation, as a fair A/B vs the from-scratch DeepUNet.

Why L4S (not HR-GLDD): Prithvi is pretrained on 6-band HLS (Blue, Green, Red,
NIR, SWIR1, SWIR2) at 10-30 m. L4S is Sentinel-2 and carries all six bands;
HR-GLDD is 3 m PlanetScope with no SWIR, so Prithvi's pretraining would not
transfer there. L4S also includes the Kodagu (India) site, so this stays
Indian-relevant. Same honest 3-way split as src/l4s.py -> directly comparable.

Encoder weights: models/prithvi/Prithvi_EO_V1_100M.pt (HF
ibm-nasa-geospatial/Prithvi-EO-1.0-100M). We drop the recomputed sincos
pos_embed and load the ViT encoder (149 tensors, 0 missing).

Band map L4S(14) -> Prithvi(6): [B2,B3,B4,B8,B11,B12] = idx [1,2,3,7,10,11].
Inputs are standardized with train-split per-band mean/std (the pretrained
conv sees ~unit-variance input, matching Prithvi's own standardization).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .l4s import CKPT_DIR, _f1_at, split_l4s_3way
from .model import FocalTverskyBCELoss

# NOTE: this is a *documented, non-adopted* experiment (see REPORT §4.2 note):
# Prithvi fine-tune scored test F1 0.669 vs the shipped DeepUNet's 0.755, so it
# was not kept. To re-run, first install deps and download the encoder:
#   uv pip install huggingface_hub einops timm safetensors
#   from huggingface_hub import hf_hub_download
#   for f in ('prithvi_mae.py','config.json','Prithvi_EO_V1_100M.pt'):
#       hf_hub_download('ibm-nasa-geospatial/Prithvi-EO-1.0-100M', f,
#                       local_dir='models/prithvi')
PRITHVI_DIR = Path("models/prithvi")
BAND_IDX = [1, 2, 3, 7, 10, 11]  # L4S 14-band -> Prithvi 6-band
IMG = 224
LAYERS = (2, 5, 8, 11)           # ViT hidden states feeding the decoder


def _load_encoder():
    # Lazy import so `import src.prithvi_l4s` works even without the weights/deps.
    if str(PRITHVI_DIR) not in sys.path:
        sys.path.insert(0, str(PRITHVI_DIR))
    from prithvi_mae import PrithviViT
    enc = PrithviViT(img_size=IMG, num_frames=1, in_chans=6,
                     patch_size=(1, 16, 16), embed_dim=768, depth=12,
                     num_heads=12, mlp_ratio=4.0)
    sd = torch.load(PRITHVI_DIR / "Prithvi_EO_V1_100M.pt", map_location="cpu",
                    weights_only=False)
    sd = {k.replace("encoder.", ""): v for k, v in sd.items()
          if k.startswith("encoder.") and "pos_embed" not in k}
    enc.load_state_dict(sd, strict=False)
    return enc


class PrithviSeg(nn.Module):
    """Prithvi ViT encoder + multi-layer convolutional decoder -> 1 logit."""

    def __init__(self, freeze_encoder: bool = False):
        super().__init__()
        self.enc = _load_encoder()
        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False
        self.freeze_encoder = freeze_encoder
        self.proj = nn.ModuleList([nn.Conv2d(768, 256, 1) for _ in LAYERS])

        def up(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

        self.dec = nn.Sequential(
            up(256, 256), up(256, 128), up(128, 64), up(64, 32))  # 14->224
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):  # x: [B,6,224,224]
        feats = self.enc.forward_features(x)
        maps = self.enc.prepare_features_for_image_model([feats[i] for i in LAYERS])
        fused = sum(p(m) for p, m in zip(self.proj, maps))  # [B,256,14,14]
        return self.head(self.dec(fused))                   # [B,1,224,224]


# ---- data ----

def _sel(img14: np.ndarray) -> np.ndarray:
    """[H,W,14] -> [6,H,W] Prithvi band order."""
    return img14[:, :, BAND_IDX].transpose(2, 0, 1).astype(np.float32)


def _compute_norm(pairs) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(6); ss = np.zeros(6); n = 0
    for ip, _ in pairs:
        with h5py.File(ip, "r") as hf:
            a = _sel(np.nan_to_num(hf["img"][:].astype(np.float32))).reshape(6, -1)
        s += a.sum(1); ss += (a * a).sum(1); n += a.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class L4SPrithvi(Dataset):
    def __init__(self, pairs, mean, std, augment=False):
        self.pairs = pairs
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        with h5py.File(ip, "r") as hf:
            x = _sel(np.nan_to_num(hf["img"][:].astype(np.float32)))
        with h5py.File(mp, "r") as hf:
            y = hf["mask"][:].astype(np.float32)[None]  # [1,128,128]
        x = (x - self.mean) / self.std
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[:, ::-1, :]
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2)); y = np.rot90(y, k, axes=(1, 2))
        x = torch.from_numpy(np.ascontiguousarray(x))
        x = F.interpolate(x[None], size=(IMG, IMG), mode="bilinear",
                          align_corners=False)[0]
        return {"x": x, "y": torch.from_numpy(np.ascontiguousarray(y))}  # y at 128


@torch.no_grad()
def _collect(model, dl, device):
    """Return flat (prob@128, gt@128) so metrics match the DeepUNet eval."""
    model.eval(); ps, gs = [], []
    for b in dl:
        x = b["x"].to(device)
        logit = model(x)  # [B,1,224,224]
        prob = torch.sigmoid(logit)
        prob = F.interpolate(prob, size=b["y"].shape[-2:], mode="bilinear",
                             align_corners=False)
        ps.append(prob.cpu().numpy().reshape(-1))
        gs.append(b["y"].numpy().reshape(-1))
    return np.concatenate(ps), np.concatenate(gs)


def train_prithvi(epochs=40, batch_size=8, num_workers=4, seed=42, warmup=3,
                  enc_lr=1e-5, dec_lr=1e-3, freeze_encoder=False) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {device}")

    tr, va, te = split_l4s_3way(seed=seed)
    print(f"L4S 3-way (same as DeepUNet): {len(tr)}/{len(va)}/{len(te)}")
    mean, std = _compute_norm(tr)

    mk = lambda p, aug: DataLoader(L4SPrithvi(p, mean, std, aug), batch_size,
                                   shuffle=aug, num_workers=num_workers,
                                   pin_memory=True, drop_last=aug)
    tr_dl, va_dl, te_dl = mk(tr, True), mk(va, False), mk(te, False)

    model = PrithviSeg(freeze_encoder=freeze_encoder).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"PrithviSeg: {ntr:.1f}M trainable (freeze_encoder={freeze_encoder})")
    criterion = FocalTverskyBCELoss(pos_weight=5.0, alpha=0.3, beta=0.7)

    enc_p = [p for p in model.enc.parameters() if p.requires_grad]
    dec_p = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("enc.")]
    groups = ([{"params": enc_p, "lr": enc_lr}] if enc_p else []) + \
             [{"params": dec_p, "lr": dec_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(epochs - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ckpt = CKPT_DIR / "best_prithvi_l4s.pt"
    best = {"f1": -1.0}; history = []
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train()
        for b in tr_dl:
            x = b["x"].to(device)
            y = F.interpolate(b["y"].to(device), size=(IMG, IMG),
                              mode="nearest")  # supervise at 224
            opt.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        Pv, Gv = _collect(model, va_dl, device)
        va = _f1_at(Pv, Gv, 0.5)
        print(f"Epoch {ep:02d}/{epochs} ({time.time()-t0:.0f}s) "
              f"| val F1 {va['f1']:.4f} P {va['precision']:.3f} R {va['recall']:.3f} "
              f"IoU {va['iou']:.3f}")
        history.append({"epoch": ep, "val": va})
        if va["f1"] > best["f1"]:
            best = dict(va, epoch=ep)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "mean": mean.tolist(), "std": std.tolist(),
                        "freeze_encoder": freeze_encoder}, ckpt)
            print(f"  -> saved best (val F1 {va['f1']:.4f}, ep {ep})")

    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model"])
    Pv, Gv = _collect(model, va_dl, device)
    sweep = [_f1_at(Pv, Gv, t) for t in np.arange(0.20, 0.81, 0.05)]
    bv = max(sweep, key=lambda s: s["f1"]); thr = bv["thr"]
    Pt, Gt = _collect(model, te_dl, device)
    t50 = _f1_at(Pt, Gt, 0.5); tf = _f1_at(Pt, Gt, thr)

    print("\n=== HELD-OUT TEST (Prithvi fine-tune, L4S real GT) ===")
    print(f"  thr 0.50        : F1 {t50['f1']:.4f} IoU {t50['iou']:.3f} "
          f"P {t50['precision']:.3f} R {t50['recall']:.3f}")
    print(f"  thr {thr:.2f} (val) : F1 {tf['f1']:.4f} IoU {tf['iou']:.3f} "
          f"P {tf['precision']:.3f} R {tf['recall']:.3f}")
    print(f"\nPRITHVI L4S test F1 = {tf['f1']:.4f} IoU {tf['iou']:.4f}  "
          f"(DeepUNet L4S = 0.755 / 0.606)")

    report = {"model": "Prithvi-EO-1.0-100M fine-tune", "dataset": "L4S",
              "best_epoch": best["epoch"], "frozen_threshold": thr,
              "val_best": bv, "test_thr050": t50, "test_frozen": tf,
              "n_test": len(te), "compare_deepunet": {"f1": 0.755, "iou": 0.606}}
    with open(CKPT_DIR / "prithvi_l4s_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(CKPT_DIR / "history_prithvi_l4s.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Artifacts -> {CKPT_DIR}/prithvi_l4s_report.json")
    return report


if __name__ == "__main__":
    train_prithvi()
