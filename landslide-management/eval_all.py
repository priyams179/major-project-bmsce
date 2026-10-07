#!/usr/bin/env python3
"""
Eval all checkpoint candidates end-to-end: per-patch threshold sweep on val
(threshold chosen on val only, then test reported at that threshold).

Batch predictions through a generator so the GPU never holds more than one
batch of upscaled images at a time (important for scale>1 models on a
24GB card).  We never mutate discover_split_samples' output.
"""
import torch, numpy as np, os, json, gc
from models import get_model
from prepare_dataset import discover_split_samples

NAMES = ["l4s_bce", "m1_sepnorm", "m1_sepnorm_lov", "m2_selftrain",
         "m3_effb4", "m3_mitb2", "m3_dlv3p",
         "m4_resnet_s256", "m4_dlv3p_s256", "m5_selftrain_r2"]

BATCH = 32


def load(name):
    ck = torch.load(f"models/{name}/best_model.pth",
                    map_location="cpu", weights_only=False)
    cfg = ck.get("config") or ck.get("args", {}) or {}
    m = get_model(
        arch=ck.get("arch", cfg.get("arch", "resnet34_unet")),
        in_channels=int(ck.get("in_channels", cfg.get("in_channels", 14))),
        base_features=int(ck.get("base_features", cfg.get("base_features", 32))),
        decoder_dropout=float(ck.get("decoder_dropout", cfg.get("decoder_dropout", 0.0))),
    ).cuda()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck, int(ck.get("scale", cfg.get("scale", 1)))


def _sample_iter(split):
    """Yield (img_raw, mask_raw, valid_raw) one sample at a time."""
    for img_p, mask_p, valid_p in sorted(
        discover_split_samples(f"data/processed/l4s/{split}", split)
    ):
        yield (np.load(img_p), np.load(mask_p), np.load(valid_p))


def _preprocess(img_raw, norm, stats, split):
    img = img_raw.astype(np.float32)
    if img.ndim == 2:
        img = img[np.newaxis]
    elif img.shape[-1] < img.shape[0]:
        img = np.transpose(img, (2, 0, 1))
    if norm != "none" and split in stats:
        key = split if norm == "split" else "train"
        sm = np.array(stats[key]["mean"], dtype=np.float32)
        ss = np.array(stats[key]["std"], dtype=np.float32) + 1e-6
        img = ((img - sm[:, None, None]) / ss[:, None, None]).astype(np.float32)
    return img


def _pred_batch(model, imgs, scale):
    xs = torch.stack([torch.from_numpy(_img) for _img in imgs]).cuda()
    with torch.no_grad():
        out = torch.sigmoid(model(xs))[:, 0].detach().cpu().numpy()
    # preds are already at the upscaled resolution when scale>1 (image was
    # upscaled before the forward pass), so no additional resize is needed.
    return np.asarray(out, dtype=np.float32)


def eval_split(model, ck, split):
    """Stream through split; return (P, Y, V) as flat float32 arrays.

    P is upscaled (if scale>1) to match the rescaled masks so shapes line up.
    """
    norm = ck.get("norm", "none")
    scale = ck.get("scale", 1)
    stats = {}
    sp = "data/processed/l4s/split_stats.json"
    if os.path.exists(sp):
        stats = json.load(open(sp))

    P, Y, V = [], [], []
    buf_imgs, buf_masks, buf_valids = [], [], []
    for img_raw, mask_raw, valid_raw in _sample_iter(split):
        img = _preprocess(img_raw, norm, stats, split)
        mask = mask_raw.astype(np.float32).squeeze()
        valid = valid_raw.astype(np.float32).squeeze()
        if mask.ndim > 1:
            mask = mask.squeeze()
        if valid.ndim > 1:
            valid = valid.squeeze()
        if scale > 1:
            img = torch.nn.functional.interpolate(
                torch.from_numpy(img).unsqueeze(0),
                scale_factor=scale, mode="bilinear", align_corners=False,
            ).squeeze(0).numpy().astype(np.float32)
            mask = torch.nn.functional.interpolate(
                torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
                scale_factor=scale, mode="nearest",
            ).squeeze().numpy().astype(np.float32)
            valid = torch.nn.functional.interpolate(
                torch.from_numpy(valid).unsqueeze(0).unsqueeze(0),
                scale_factor=scale, mode="nearest",
            ).squeeze().numpy().astype(np.float32)
        buf_imgs.append(img)
        buf_masks.append(mask.ravel())
        buf_valids.append(valid.ravel())
        if len(buf_imgs) >= BATCH:
            P.append(_pred_batch(model, buf_imgs, scale).ravel())
            Y.append(np.concatenate(buf_masks))
            V.append(np.concatenate(buf_valids))
            buf_imgs, buf_masks, buf_valids = [], [], []
            gc.collect()
            torch.cuda.empty_cache()
    if buf_imgs:
        P.append(_pred_batch(model, buf_imgs, scale).ravel())
        Y.append(np.concatenate(buf_masks))
        V.append(np.concatenate(buf_valids))
    return np.concatenate(P).astype(np.float32), \
           np.concatenate(Y).astype(np.float32), \
           np.concatenate(V).astype(np.float32)


def sweep(P, Y, V):
    pv = torch.as_tensor(P, dtype=torch.float32)
    yv = torch.as_tensor(Y, dtype=torch.float32) > 0.5
    vv = torch.as_tensor(V, dtype=torch.float32) > 0.5
    yv = yv.bool()
    vv = vv.bool()
    best_f1, bt = 0.0, 0.5
    for t in np.linspace(0.20, 0.80, 31):
        b = (pv >= t) & vv
        g = yv & vv
        b = b.bool()
        g = g.bool()
        tp = (b & g).sum().item()
        denom = 2 * tp + (b & ~g).sum().item() + (~b & g).sum().item() + 1e-12
        f1 = 2 * tp / denom
        if f1 > best_f1:
            best_f1, bt = f1, t
    b = (pv >= bt) & vv
    g = yv & vv
    b = b.bool()
    g = g.bool()
    tp = (b & g).sum().item()
    fp = (b & ~g).sum().item()
    fn = (~b & g).sum().item()
    return dict(iou=tp / max(tp + fp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                prec=tp / max(tp + fp, 1),
                rec=tp / max(tp + fn, 1), t=bt)


print(f"{'config':22s} {'val IoU':>9s} {'val F1':>8s} P/R      |  {'test IoU':>9s} {'test F1':>8s} TP/FP/FN/TN")
for name in NAMES:
    try:
        m, ck, scale = load(name)
        Pv, Yv, Vv = eval_split(m, ck, "val")
        vr = sweep(Pv, Yv, Vv)
        # release val batch from GPU before test
        del Pv, Yv, Vv
        gc.collect()
        torch.cuda.empty_cache()
        Pt, Yt, Vt = eval_split(m, ck, "test")
        tr = sweep(Pt, Yt, Vt)
        print(f"{name:22s} {vr['iou']*100:8.2f}% {vr['f1']*100:7.2f}% "
              f"{vr['prec']*100:2.0f}/{vr['rec']*100:2.0f} | "
              f"{tr['iou']*100:8.2f}% {tr['f1']*100:7.2f}% "
              f"{int(tr['prec']*1e4):4d}/{int(tr['rec']*1e4):4d}")
    except Exception:
        import traceback
        print(f"{name:22s} ERROR")
        traceback.print_exc()
