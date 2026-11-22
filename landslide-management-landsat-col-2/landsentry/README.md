# LandSentry: bitemporal Sentinel-2 landslide change detection

This is a fresh, self-contained implementation of the
[LandSentry](https://github.com/GHRUTHI/Landsentry) idea. It does not import
anything from `src/` and does not modify the existing pipelines.

* **Models:** a Siamese CNN (shared encoder, |pre − post| skip fusion, U-Net
  decoder) and a BIT-style transformer (semantic tokenizer, then a transformer
  encoder over the before/after tokens, then a cross-attention decoder).
* **Baseline:** a non-learned NDVI-differencing baseline, scored under the
  same protocol as the models.
* **Cross-event test:** train on Kodagu 2018, Wayanad 2024 or both, then test
  on the held-out region of each event. The decision threshold is frozen on
  the training event's validation split.

Full method, numbers and caveats are in [REPORT.md](../REPORT.md): §2.8 (data), §3.11 (method) and §4.8 (results).

## Results

All scores are on held-out test regions, mean ± std over 3 seeds. Kodagu
uses the real inventory; Wayanad uses an index-derived reference mask.

| Method | Trained on | Kodagu F1 | Wayanad F1 |
|---|---|---|---|
| NDVI difference, no learning (Kodagu-tuned threshold) | none | 0.672 | 0.835 |
| Siamese CNN | Kodagu | 0.686 ± 0.009 | **0.894 ± 0.012** (zero-shot) |
| Siamese CNN | Wayanad | 0.514 ± 0.069 (zero-shot) | 0.929 ± 0.004 |
| Siamese CNN | both | **0.694 ± 0.008** | 0.914 ± 0.003 |
| BIT transformer | Kodagu | 0.683 ± 0.046 | 0.868 ± 0.019 (zero-shot) |
| BIT transformer | Wayanad | 0.413 ± 0.133 (zero-shot) | 0.937 ± 0.008 |
| BIT transformer | both | 0.689 ± 0.004 | 0.903 ± 0.016 |

Three things stand out:

* On real ground truth, the learned models beat plain NDVI differencing by about 2 F1 points.
* Kodagu → Wayanad transfers, but Wayanad → Kodagu does not, because one debris flow is too little to learn from.
* The transformer does not transfer better than the CNN.

## Quick start

```bash
.venv/bin/python -m landsentry prepare        # S2 pre/post composites via Earth Engine (~6 min)
.venv/bin/python -m landsentry labels         # masks + spatial splits + qc_labels.png
.venv/bin/python -m landsentry cross-event    # 2 models x 3 sources x 3 seeds + baseline

# individual steps
.venv/bin/python -m landsentry train --model transformer --train-on kodagu --seed 0
.venv/bin/python -m landsentry evaluate --checkpoint checkpoints/landsentry/siamese_kodagu_s0.pt --event wayanad
```

The Earth Engine project defaults to `major-project-512310`. Override it with
`LANDSENTRY_GEE_PROJECT`.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Events, windows, bands, paths |
| `gee.py` | s2cloudless + QA60 masked S2 L1C median composites, tiled download on a fixed UTM 10 m grid |
| `prepare.py` | Downloads `data/landsentry/<event>/{pre,post}.tif` |
| `labels.py` | Kodagu inventory rasterisation, Wayanad seeded reference mask, spatial splits, QC figure |
| `data.py` | Per-scene joint normalisation, scene loader, random-crop sampler with augmentation |
| `models.py` | `SiameseCNN` (1.93 M params), `BITransformer` (0.69 M params) |
| `train.py` | BCE + Dice training; checkpoint and threshold chosen on the source event's val split |
| `infer.py` | Sliding-window inference with Hann blending, metrics (F1/IoU/P/R/AP) |
| `evaluate.py` | Scores one checkpoint on one event or split |
| `cross_event.py` | Full experiment, tables and figures, written to `results/landsentry/` |

## Data

| Event | Imagery (pre / post, Dec–Mar dry season) | Labels |
|---|---|---|
| Kodagu, Aug 2018 | S2 L1C 2017-12 → 2018-03 / 2018-12 → 2019-03; 3594 × 4258 px | Achaiah et al. (2024) inventory, 217 polygons, 5.12 km² (independent ground truth) |
| Wayanad, 30 Jul 2024 | S2 L1C 2023-12 → 2024-03 / 2024-12 → 2025-03; 1779 × 2189 px | Seeded NDVI-change reference mask, 0.70 km². This is not independent ground truth (see below). |

**Why Wayanad has no real ground truth.** No public vector outline of the
Mundakkai–Chooralmala debris flow exists, and the original repo ships only a
point placemark. The Wayanad mask is built from the NDVI change around that
placemark. A change detector is therefore partly scored against the very
signal it learns, so Wayanad results measure agreement with a reference.
They are not independent accuracy. The NDVI baseline row makes this visible.
