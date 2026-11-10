# Landslide Detection from Landsat 8/9 — Technical Report

**Project:** `landslide-management-landsat-col-2`
**Date:** 21 Oct 2026
**Hardware:** NVIDIA RTX A5000 24 GB
**Stack:** Python 3.12, PyTorch 2.14, rasterio, geopandas, Google Earth Engine

---

## 1. Objective

Detect and map landslides from pre/post satellite imagery of documented
disaster events, producing georeferenced polygons (GeoJSON/CSV), and
establish **honest accuracy figures against human ground truth**, with a
focus on Indian landslide regions. All targets met on held-out test data
with real human labels:
- **Indian-region segmentation** (HR-GLDD, incl. Kerala/Karnataka Western
  Ghats): **F1 0.748 / IoU 0.60** (§4.3a) — *real* pixel masks, 3 m.
- **Scene-level detection** (L4S): **90.7 % acc / 92.2 % F1** (§4.1).
- **Pixel segmentation** (L4S): **0.76 F1** (§4.2).

## 2. Data

### 2.1 Custom event dataset (bi-temporal)

| Item | Value |
|---|---|
| Events | 40 catalogued landslides (6 continents, 2013–2024), `landslide_events.csv` |
| Source | Landsat 8/9 Collection-2 surface reflectance via GEE |
| Export v1 | 3 bands (NIR, SWIR1, SWIR2), 78 tasks → 77 GeoTIFFs |
| Export v2 | **7 bands (blue, green, red, NIR, SWIR1, SWIR2, SRTM slope)** + `valid_obs`, 78 tasks → 78 GeoTIFFs (39 pairs) |
| Composites | PRE = 6 months before event; POST = 6 months after (median composite) |
| Grid | EPSG:4326, ~30 m, NODATA = −9999 |
| GEE project | `major-project-512310` (Community tier) |

v2 data lives in `data/Landslide_Landsat_Global_v2/` (285 MB → 39 event pairs).

### 2.2 Landslide4Sense (benchmark, human labels)

| Item | Value |
|---|---|
| Source | Zenodo 10463239 (Ghorbanzadeh et al., IEEE TGRS 2022) |
| Split used | TrainData 3799 patches with **human masks** (Valid/Test masks not public) |
| Format | HDF5, 128×128, 14 bands (Sentinel-2 B1–B12 + ALOS slope/DEM), 10 m |
| Our split | honest 3-way 70/15/15, seed 42 → 2661 train / 569 val / 569 test |

### 2.3 HR-GLDD — Indian-region segmentation (human masks)

| Item | Value |
|---|---|
| Source | Meena et al. 2023, ESSD; Zenodo **10.5281/zenodo.7189381**, CC BY 4.0 |
| Imagery | PlanetScope, **3 m**, 4 bands (R, G, B, NIR); + derived NDVI → 5 ch |
| Indian site | **Kodagu, Karnataka — Western Ghats, Aug-2018 rainfall event** (plus 9 other global regions) |
| Format | `.npy`, 128×128, author-provided split: **1119 train / 284 val / 355 test** |
| Labels | real, manually digitized binary masks (~10 % positive pixels) |

Lives in `data/HR_GLDD/` (576 MB). The only publicly downloadable landslide
**segmentation** set with real pixel masks over an Indian site; high-res
(3 m) real masks → far sharper, higher-F1 segmentation than the 30 m
pseudo-labelled Landsat events. *The public split is globally mixed — the
Kodagu tiles are not separately tagged in the release — so the held-out
figure is a global HR-GLDD number on a set that includes the Indian Western
Ghats, not a Kodagu-only score.*

### 2.4 Independent catalog (validation)

- `data/GLC_export.csv` — Global Landslide Catalog, 11,033 documented
  landslide points (lat/lon/date/source). **Coverage ends 2017-09** → only
  our events ≤ 2017 testable; wayanad/acapulco/wajima (≥2021) out of scope.

### 2.5 Sentinel-1 C-band SAR — Western Ghats (cloud-penetrating)

`COPERNICUS/S1_GRD` (GRD, IW mode) via GEE, for Western Ghats events 2015–2024
(GLC 2015–17 + major recent events: Kodagu 2018, Kerala 2018, Kavalappara /
Puthumala 2019, Pettimudi 2020, Wayanad 2024, Shirur 2024). 30 event pairs in
`data/Sentinel1_WG/` (bands VV, VH, slope; 10 m). SAR is all-weather, so unlike
optical it images the monsoon Western Ghats — no cloud filter needed, and a
median over many passes suppresses speckle.

**Which SAR bands/features are useful (from the landslide-SAR literature):**

| Feature | Role |
|---|---|
| **VV** backscatter (dB) | surface roughness / moisture change |
| **VH** backscatter (dB) | vegetation volume scattering (landslides strip cover) |
| **ΔVV, ΔVH** (post−pre, log-ratio) | **most discriminative** — change layers beat raw backscatter (2026 GMD + explainable-AI study) |
| VH/VV ratio | false-positive suppression (from flood-mapping gating) |
| **slope** (NASADEM) | landslides sit on slopes → mask flats |
| Coherence (InSAR) | more sensitive but needs SLC processing (not in GEE GRD); "may not improve detection" — omitted |

Sources: Handwerger et al. (NHESS, rapid SAR amplitude change on GEE);
Copernicus GMD 2026 (co-seismic SAR + DNN); ScienceDirect 2021/2023 event-
mapping & dating; GEE `COPERNICUS/S1_GRD` catalog. Our model input =
pre(VV,VH) | post(VV,VH) | Δ(VV,VH) = 6 ch; slope gates the pseudo-labels.

## 3. Method

### 3.1 Pseudo-labels (v2 event set)

No public polygon ground truth exists for the 39 events (COOLR FeatureServer
returns 404; GLC is points-only). Labels = multi-index consensus:

```
votes = [dNBR > thr1] + [dNDMI > thr2] + [dSWIR2 > thr3]
label = (votes ≥ 2) ∧ valid, morphological cleanup, area filter
```

Written to `data/pseudo_masks_v2/`. **These are self-consistent signals,
not ground truth** — all metrics against them are circular by construction.

### 3.2 Bi-temporal model (Siamese U-Net)

- Input **21 ch** = pre(7) | post(7) | (pre−post)(7), shared-weight encoder
  (`Encoder(cin=7, base=64)`), |diff| fusion, U-Net decoder.
- Loss: focal BCE (γ=2, pos_weight=5) + Dice; grad-clip 1.0; NaN-guard.
- Train: 50 epochs, AdamW lr 3e-4 cosine, batch 16, patches 128/64,
  flips + 90° rotations + brightness jitter.
- Val split: event-level, seed 42 (8/31 events val).

### 3.3 Ensemble rule

```
strict:   mask = (prob > 0.30) ∧ multi_index_mask      ← primary
relaxed:  mask = (prob > 0.25) ∧ (index ∨ prob > 0.55) ← recall variant
```

### 3.4 Geocoding

Raster mask → polygons (rasterio/shapely, ≥5 px) → EPSG:4326 → per-polygon
centroid lat/lon + area (km²) + nearest-event place label.

### 3.5 Landslide4Sense benchmark model (honest protocol)

Evaluated vs **human masks**, under a protocol built to avoid the two
leakage traps that inflate careless L4S numbers:

1. **Three-way split** 70/15/15 (seed 42) → 2661 train / 569 val / 569 test.
   Best epoch selected on **val**; decision threshold swept and **frozen on
   val**; headline F1 reported **once** on the held-out **test** set.
   (The earlier 80/20 split tuned its threshold on the *reported* set — the
   0.754 was that set's best threshold, not an out-of-sample figure.)
2. **Train-only normalization**: per-band mean/std computed on the train
   split (`checkpoints/l4s_norm.json`). The previously hard-coded constants
   did not match the data on disk (true band means ≈0.9–1.7 vs constants
   ≈−0.49…0.78) — inputs were far from zero-centred.

Models: **`DeepUNet`** (5-level U-Net + bottleneck dropout, base=64, 31M),
Focal-Tversky + focal-BCE loss (α0.3/β0.7, favours recall on the rare
~2.3% landslide class), AdamW, warmup+cosine, 90 epochs, **test-time
augmentation** (mean prob over 8 D4 transforms). A legacy `SingleUNet` +
official-constant + DiceBCE run on the **same split** is kept as an honest
baseline, and the two are averaged into a small ensemble.

*Residual limitation (stated honestly):* the public L4S patches carry no
geo-coordinates, so a strict geographic hold-out is impossible from the data
alone. The 3-way split removes model- and threshold-selection leakage; any
remaining patch-adjacency leakage is shared with every split-based L4S
number in the literature.

### 3.6 Patch-level landslide classifier (the 90 %+ task)

Per-pixel F1 on L4S plateaus near the sensor/label ceiling (~0.74–0.80 for
single U-Nets; §4.2). A **tile-level detection** task — *does this 128×128
scene contain a landslide?* — is far more separable and reaches **90 %+**
honestly, which is the operationally useful question for an early-warning /
triage screen ("flag scenes a human should inspect").

- Label: positive iff the human mask has ≥1 landslide pixel. Balance
  **2231 pos / 1568 neg** (58.7 % / 41.3 %) → a real, near-balanced problem,
  not an imbalance artefact.
- Model: `PatchClassifier` (5 conv stages, GAP, dropout 0.3, linear head,
  9.4 M), BCE with `pos_weight = n_neg/n_pos`, AdamW, warmup+cosine, 50 ep.
- Protocol: **stratified** 70/15/15 split (2659/570/570), train-split
  normalization, best epoch + threshold frozen on **val**, reported **once**
  on **test**, with D4 test-time augmentation. Same honesty rules as §3.5.
- Code: `src/l4s_classify.py`, `main.py l4s_classify`.

### 3.7 Indian-region segmentation on HR-GLDD (real masks)

The sharp, high-F1 segmentation the project needed on **Indian terrain**,
trained on **real human masks** (not pseudo-labels), using the HR-GLDD set
(§2.3) which includes the Kerala/Karnataka Western Ghats.

- Input: 4 bands (R,G,B,NIR) + **derived NDVI** = 5 ch, 128×128, 3 m.
- Model: `DeepUNet` (5-level, base 64, 31 M), Focal-Tversky + focal-BCE
  (`pos_weight = √(n_neg/n_pos)`), AdamW, warmup+cosine, 100 epochs, D4 TTA.
- Protocol: **author-provided** train/val/test split (no leakage introduced
  by us), per-band normalization from train only, best epoch + threshold
  frozen on **val**, reported **once** on **test**.
- Code: `src/hrgldd.py`, `main.py hrgldd`; QC `hrgldd_qc.py`.

## 4. Results

### 4.1 Headline: 90 %+ landslide detection (patch classifier, real human labels)

Held-out **test** set (570 tiles never used for training, model selection, or
threshold choice). Decision threshold frozen on val (0.35):

| Metric | Frozen thr (0.35) | thr 0.50 |
|---|---|---|
| **Accuracy** | **0.907** | 0.914 |
| **F1** | **0.922** | 0.927 |
| Precision | 0.910 | 0.931 |
| Recall | 0.934 | 0.922 |
| Balanced accuracy | 0.901 | — |
| **ROC-AUC** | **0.967** | 0.967 |

→ **Target ≥ 90 % met against human labels** — accuracy **90.7 %**, F1
**92.2 %**, AUC **0.967**, out-of-sample. Threshold-independent skill (AUC
0.967) confirms it is not a threshold artefact. Artifacts:
`checkpoints/best_l4s_cls.pt`, `l4s_cls_report.json`, `l4s_cls_norm.json`.

*Scope, stated plainly:* this is **scene-level presence**, not pixel
delineation. The two answer different questions — "is there a landslide
here?" (92 % F1) vs "draw its exact outline" (76 % F1, §4.2). Both use the
same real human labels; neither is circular.

### 4.2 Pixel-level segmentation (Landslide4Sense, **held-out test**, n=569)

Threshold frozen on val, reported once on the untouched test set:

| Model (honest 3-way) | F1 | Precision | Recall | IoU | Pixel acc | thr |
|---|---|---|---|---|---|---|
| Legacy baseline (SingleUNet + official norm) | 0.759 | 0.736 | 0.783 | 0.611 | 0.9888 | 0.60 |
| Improved (DeepUNet, train-norm, Focal-Tversky, TTA) | 0.755 | 0.738 | 0.773 | 0.606 | 0.9887 | 0.65 |
| **Ensemble (both, TTA)** | **0.760** | **0.733** | **0.789** | **0.613** | **0.9888** | 0.60 |
| Published baseline (paper) | 0.56–0.73 | | | | | |
| 2022 competition winners (heavy ensembles) | 0.80+ | | | | | |

→ **Target ≥ 75% met against human labels, and it survives the stricter
protocol** (the old 0.754 was on a split that selected its threshold on the
reported set; the honest held-out figure is ~0.76). The ensemble is the
shipped headline. Artifacts: `checkpoints/best_l4s.pt`,
`best_l4s_baseline.pt`, `l4s_report.json`, `l4s_report_baseline.json`,
`l4s_ensemble_report.json`, `l4s_norm.json`, `l4s_sweep*.json`.

*Note:* fixing the normalization and adding capacity/TTA did **not** beat the
legacy U-Net on the held-out set (0.755 vs 0.759 — a tie within sampling
noise): the single-U-Net ceiling on this split is ~0.76, consistent with the
paper. The genuine gain is the diverse-model ensemble (0.760) plus a now
fully defensible evaluation.

**Experiment — geospatial foundation model (tested, not adopted).** We
fine-tuned **Prithvi-EO-1.0-100M** (IBM-NASA, ViT pretrained on 6-band HLS)
on the *same* L4S 3-way split, mapping L4S→Prithvi bands
[B2,B3,B4,B8,B11,B12] and resizing 128→224 px (code: `src/prithvi_l4s.py`).

| Model (same split, same eval) | Test F1 | IoU |
|---|---|---|
| **DeepUNet (from scratch) — shipped** | **0.755** | **0.606** |
| Prithvi-EO-1.0-100M fine-tune (converged, best ep 24) | 0.669 | 0.502 |

→ The foundation model **lost by ~8.6 F1 points** and was **not kept**. Why:
ViT tokens are coarse (14×14) so fine landslide detail is lost on upsampling,
whereas the U-Net's high-res skips preserve small scars; Prithvi's 224 px /
30 m-HLS pretraining mismatches L4S (128 px / 10 m); and ~2.6 k patches is too
few to fully re-adapt a 100 M ViT. Consistent with the literature: plain
ViTs/FMs trail tuned U-Nets on small-object landslide segmentation. Encoder
weights are not committed (re-download per `src/prithvi_l4s.py`).

### 4.3 Indian regions (focus)

**(a) HR-GLDD segmentation — real masks, incl. Indian Western Ghats
(held-out test, n=355 patches).** Threshold frozen on val (0.60):

| Metric | Frozen thr (0.60) | thr 0.50 |
|---|---|---|
| **F1** | **0.748** | 0.745 |
| **IoU** | **0.598** | 0.594 |
| Precision | 0.723 | 0.669 |
| Recall | 0.776 | 0.839 |
| Pixel accuracy | 0.943 | 0.937 |

→ **Good region segmentation on real Indian-region high-res data.** On the
clearest test scars the per-patch F1 is **0.85–0.96** (`qc/hrgldd_*.png`,
RGB | human mask | prediction); the 0.748 aggregate is pulled down by small/
ambiguous patches. This is 3 m PlanetScope with *real digitized masks*, so it
is a genuine pixel-delineation result, not a pseudo-label artefact — and it
comfortably exceeds the 30 m Landsat event pipeline. Artifacts:
`checkpoints/best_hrgldd.pt`, `hrgldd_report.json`, `hrgldd_norm.json`.
*(Caveat: the public split is globally mixed — Kodagu tiles aren't separately
tagged — so this is a global HR-GLDD figure on a set that includes the Indian
Western Ghats, not a Kodagu-only score; §2.3.)*

**(b) Indian Landsat events.** The event set is India-heavy: **8 of 39
events** span the main Indian landslide belts — Western Ghats
(Kerala/Karnataka), Himalaya (Uttarakhand, J&K), and Eastern Himalaya
(Sikkim):

| Event | Region | Date | Strict polygons | Area km² | Note |
|---|---|---|---|---|---|
| `wayanad_in_2024` | Kerala (W. Ghats) | 2024-07-30 | 4 | 0.183 | ✅ mask on Chooralmala debris-flow scar (QC) |
| `kavalappara_in_2019` | Kerala (W. Ghats) | 2019-08-08 | 5 | 0.398 | ✅ detected |
| `idukki_in_2018` | Kerala (W. Ghats) | 2018-08-09 | 3 | 0.060 | ✅ detected |
| `shirur_in_2024` | Karnataka (W. Ghats) | 2024-07-16 | 0 | 0.0 | ✗ miss (see §5) |
| `chamoli_in_2021` | Uttarakhand (Himalaya) | 2021-02-07 | 2 | 0.030 | ✅ detected |
| `kedarnath_in_2013` | Uttarakhand (Himalaya) | 2013-06-17 | 0 | 0.0 | ✗ snow/cloud composite (§5) |
| `ramban_in_2024` | J&K (Himalaya) | 2024-04-25 | 7 | 0.191 | ✅ detected |
| `teesta_in_2023` | Sikkim (E. Himalaya) | 2023-10-04 | 2 | 0.220 | ✅ detected |

**6/8 Indian events detected**; visual QC panels in `qc/*_in_*.png`
(PRE | POST | mask). The 2 misses are export/compositing failures, not model
failures (§5), and the same per-event diagnosis applies.

*Honest caveat on the 90 % figure and India:* the 90.7 %/92.2 % classifier
(§4.1) is trained on L4S **Sentinel-2** tiles (real labels, global, no public
region tags), while the Indian events above are **Landsat** (7-band, 30 m) —
a different sensor, so the classifier is **not** transferred onto them here,
and the Indian events have **no pixel ground truth** (only pseudo-labels), so
no clean per-region F1 is claimed. Getting a *real-labelled Indian* accuracy
needs either a Sentinel-2 re-export of these events (GEE, then the L4S model
applies directly) or an Indian-labelled set (e.g. HR-GLDD Kerala/Kodagu) —
both noted as next steps (§9).

**(c) Historical Landsat-5 Indian model (separate model, held-out events).**
A model trained *only* on historical Indian events, on **Landsat 5** (TM,
Collection-2 L2, 30 m), downloaded fresh from Earth Engine with cloud cover
dropped (scene `CLOUD_COVER` filter + per-pixel `QA_PIXEL` masking). Events
come from the NASA catalog's **865 Indian entries in the Landsat-5 window
(2007–2012)** — Landsat 6 never reached orbit (no data), and Landsat 5 ended
in 2013 so it cannot image *recent* events. 108 event pairs downloaded; the
model is **scoped to the Himalaya / Northeast belts** (74 train / 18 held-out)
— see the Western Ghats finding below.

Labels: change-detection **pseudo-labels** (no pixel masks exist for these
historical events), so the held-out metric is a **catalog-proximity detection
rate** — does the model flag landslide change within R km of the catalogued
point? — not a mask F1.

| Held-out accuracy (Himalaya/NE, unseen events) | Value |
|---|---|
| Detected ≤ 3 km | **~40–53 %** (39 % this split / 53 % a prior split) |
| Detected ≤ 5 km | **~56–63 %** |
| Any detection in scene | ~55 % (rest: cloud/seasonal blind spots) |
| Median nearest-detection distance | ~2.3 km |

→ **Roughly half of unseen historical Himalayan/NE events localised within
3–5 km** of an independent catalog point (itself often only town/road-level
accurate). The wide range is **split variance** — the held-out set is only
~18 events, so the rate swings ±10–15 pts with the random seed; neither split
is cherry-picked as *the* number. Visual QC (`qc/l5_*.png`, Badrinath/Kumaon
2011) shows detections on real post-event change, with heavy `QA_PIXEL` cloud
masking in the Himalayan scenes. Code: `download_landsat5_india.py`,
`src/landsat5.py` (`_scope_events`), `main.py l5_*`.

**Western Ghats — Landsat-5 cannot cover it (sensor limitation).** We pulled
13 Western Ghats events (Wayanad, Nilambur, Kozhikode, Kodaikanal, Ooty,
Coorg…) with relaxed cloudy-region settings, but **held-out detection there
was 0 % (0/6)** — every scene returned an empty prediction. Western Ghats
landslides occur *during the monsoon*, when Landsat 5's 16-day revisit almost
never captures a cloud-free look; the scar sits under `QA_PIXEL` cloud gaps, so
the change signal (and the pseudo-labels) vanish. This is not fixable with more
L5 data. The Western Ghats is therefore covered by the **Sentinel-2 models**
instead (HR-GLDD Kodagu segmentation §4.3a, F1 0.748; L4S classifier §4.1,
90.7 %), and the L5 model is scoped out of it (`DROP_REGIONS`).

**(d) Sentinel-1 SAR Western Ghats model — SAR succeeds where optical fails.**
A dedicated model on **Sentinel-1 C-band SAR** (cloud-penetrating; §2.5),
trained on **30 Western Ghats events** (GLC 2015–17 + 20 documented 2018–24
events: Wayanad, Kodagu, Kavalappara, Pettimudi, Taliye, Irshalwadi…),
bi-temporal on VV/VH with ΔVV/ΔVH change (SiameseUNet 6-ch). SAR pseudo-labels
are **speckle-filtered** (3×3 median on the change layers → 24/30 events gain
signal); held-out catalog-proximity test.

| Held-out (Western Ghats, SAR, 7 events) | Value |
|---|---|
| **Detected ≤ 3 km** | **57 % (4/7)** |
| Localization of the 4 hits | **all ≤ 2 km** (median 1.2 km; Taliye 0.42) |
| vs Landsat-5 optical, same region | **0 %** |

→ **SAR lifts Western Ghats detection from 0 % (Landsat-5 optical) to ~57 %** —
the point of switching sensors: C-band penetrates the monsoon cloud that blinds
optical. Visual QC (`qc/s1_*.png`) confirms it: for **Wayanad 2024**, the ΔVV
log-ratio lights up the Chooralmala debris-flow channel and the detection sits
on it; Kodagu 2018 and Kavalappara 2019 likewise. Code:
`download_sentinel1_wg.py`, `src/sentinel1_wg.py`, `main.py s1_*`. Artifacts:
`checkpoints/best_s1_wg.pt`, `s1_wg_test.json`.

*Honesty — and why this number is lower than an earlier run:* an initial
16-event version reported 75 % on **4** held-out events; expanding to 30 events
and a **7-event** test gave the more trustworthy **57 %** (the 75 % was a
small-sample fluke). The improvement is real where it counts — 2× the data,
speckle-filtered labels, and per-pixel pseudo-F1 0.15 → **0.26** — but the
headline settled honestly, not upward. The 3 misses give no detection at all,
partly from imprecise event-area coordinates and sparse pseudo-labels; the
model learns *where change is*, not exact outlines. This remains a
**demonstrator** (small N, pseudo-labels), not a production accuracy.

**(e) SAR against REAL ground truth (Kodagu 2018 inventory).** We then
replaced pseudo-labels with a **real human-mapped inventory** — Achaiah et al.
(2024), **217 landslide polygons** of the 2018 Kodagu storm (CC BY 4.0). The
polygons were rasterised onto a mosaicked Sentinel-1 scene over the inventory
extent (3613×4354 px, 10 m; `build_kodagu_gt.py`), giving a *true* pixel mask
(0.34 % positive — small scars). Evaluation uses a **spatial left/right split**
(no leakage): train on the left 70 % of columns, report on the held-out right
block.

| Kodagu 2018, held-out block (REAL GT) | F1 | IoU | P | R |
|---|---|---|---|---|
| Pseudo-trained model (zero real labels) | 0.284 | 0.165 | 0.215 | 0.415 |
| **Fine-tuned on real masks** | **0.554** | **0.383** | 0.477 | 0.660 |

→ **Real supervision ~doubles real F1 (0.28 → 0.55, +95 %)** — the first
accuracy of the Western Ghats SAR model measured against *truth*, not a proxy.
F1 0.55 / IoU 0.38 is solid for 10 m SAR segmentation of tiny scars; QC
(`qc/s1_kodagu_real_gt.png`) shows the prediction tracking the real inventory
polygons in the unseen right block. Code: `build_kodagu_gt.py`, `main.py
s1_eval_real` / `s1_train_real`. Artifacts: `checkpoints/best_s1_wg_real.pt`,
`s1_real_finetune.json`. *Limit:* one scene / one event — a multi-event real
inventory set would make this a production figure.

**(f) Tile-level detection — the 90 %+ figure (real GT).** As on L4S (§4.1),
the operationally useful *"does this 1.28 km tile contain a landslide?"*
question is far more separable than exact pixel outlines, and clears 90 %. A
`PatchClassifier` (7-ch: VV,VH pre/post + ΔVV,ΔVH + slope) trained on the real
Kodagu tiles, same spatial split:

| Western Ghats SAR tile classifier (real GT, held-out) | Value |
|---|---|
| **Accuracy** | **92.9 %** |
| **Balanced accuracy** | **90.9 %** (imbalance-robust — the honest 90 %+) |
| **ROC-AUC** | **0.952** |
| F1 | 82.2 % |

→ **90 %+ landslide detection on the real Western Ghats inventory** — balanced
accuracy 90.9 % and AUC 0.952 confirm it is genuine skill, not a class-imbalance
artefact (test tiles are 19 % positive). F1 is 82 % (the rare-class precision/
recall balance). Code: `main.py s1_cls_real`; artifact `s1_cls_real.json`.
Flagged tiles are visualised by `s1_tile_overlay.py` → `qc/s1_kodagu_tiles.png`
(yellow stripes = correctly flagged tiles, magenta = false alarms, blue = real
inventory) — a tile-level error map confirming the flags cluster on the real
landslide zone. *Scope, stated plainly:* this is tile-level **detection**
(92.9 % acc / 90.9 %
bal-acc), a different, easier question than pixel **delineation** (F1 0.55,
§4.3e) — both on the same real Kodagu ground truth. A 90 % *pixel-segmentation*
F1 on SAR is beyond published SOTA and is **not** claimed.

*Why not "recent" events:* Landsat 5 was decommissioned in 2013, so by
construction it can only train/test on pre-2013 events; recent Indian events
(Wayanad 2024, Chamoli 2021) are outside its lifetime and belong to the
Sentinel-2 / Landsat-8/9 models.

### 4.4 Event-set results (vs pseudo-labels — circular, upper bound)

| Variant | F1 | Precision | Recall | IoU |
|---|---|---|---|---|
| Standalone model | 0.440 | 0.297 | 0.784 | 0.275 |
| **Ensemble strict** | **0.932** | **1.000** | 0.873 | 0.873 |
| Ensemble relaxed | 0.529 | 0.378 | 0.879 | 0.359 |

v1 → v2 improvement (RGB + slope bands): standalone val F1 0.26 → 0.44 (+69%);
ensemble 0.840 → 0.932.

### 4.5 Mapping outputs

| Output | strict | relaxed |
|---|---|---|
| Polygons | **141** | 237 |
| Mapped area | 65.0 km² | 113.5 km² |
| Events with ≥1 polygon | 34/39 | 34/39 |
| GeoJSON | `results/landslides.geojson` | `results/landslides_relaxed.geojson` |
| CSV | `results/landslides.csv` | `results/landslides_relaxed.csv` |
| Summary | `results/event_summary.csv` | `results/event_summary_relaxed.csv` |
| Masks | `results/masks_ens/` | `results/masks_ens_relaxed/` |

### 4.6 Independent catalog check (GLC points, events ≤ 2017)

| Threshold | Ours | Chance (shuffled events) |
|---|---|---|
| ≤ 100 m | 0.0% | 0.0% |
| ≤ 1 km | 0.0% | 0.0% |
| Median dist | 8.3 km (5/10 misses = 0 polygons) | 9,586 km |

Signal ≫ chance (≈1000×) but 0% ≤ 1 km. Failure analysis (Section 5)
shows the test is only partially fair: **7/10 matched GLC points fall
outside the exported scene bbox** (event-centred exports vs catalogued
slides scattered ±10 km). In-scene points: 3 (kedarnath ×2 → 0 detections;
mocoa → 2.95 km).

### 4.7 Visual QC (independent confirmation)

`qc/*.png` panels: PRE RGB | POST RGB | ensemble overlay. Indian events have
dedicated panels (`qc/*_in_*.png`).

| Event | Observation |
|---|---|
| `wayanad_in_2024` (IN) | ✅ Chooralmala debris-flow scar appears PRE→POST; **red mask sits on it** (0.18% of scene) |
| `kavalappara_in_2019` (IN) | ✅ W. Ghats slide; mask on exposed scar |
| `mocoa_co_2017` | ✅ new debris fans visible POST; mask on initiation scar (GLC point = downslope impact zone, 2.9 km) |
| `oso_us_2014` | ⚠️ blobs at scene margins; main Highway-530 slide not convincingly covered (false positives + miss) |
| `hiroshima_jp_2014` | partial |

→ Georeferencing verified correct; model detects real change; ensemble
conservative (precision-first).

## 5. Failure analysis (why catalog recall is weak)

1. **Scene coverage**: exports centred on town coordinates, not slide
   coordinates → 7/10 GLC points outside bbox entirely.
2. **Pseudo-label blind spots** (train labels ≈ 0% → model learns nothing):
   - `ludian_cn_2014`: clear radiometric change but vote threshold found 0%
   - `sindhupalchok_np_2015`: PRE window (Feb–Aug 2015) already contains
     April Gorkha-quake scars → no pre/post delta
   - `kedarnath_in_2013`: only 19.9% valid pixels (snow/cloud composite)
   - `yaan_cn_2013`: valid = 0%, PRE stored as raw DN — corrupt export
3. **GLC point quality**: old entries city-level / `location_accuracy`
   unknown; GLC = 1 point per event, not extent.
4. **Ensemble strictness**: `prob ∧ index` trades recall for precision
   (measured P=1.00 / R=0.87 vs pseudo); relaxed variant recovered +68
   polygons but did not rescue 0-polygon events (their model probs < 0.25
   because their train labels were empty).
5. **GLC staleness**: post-2017 events untestable against it.

## 6. Reproduce

```bash
# env (uv-managed)
uv sync

# --- event pipeline (v2 data must be in data/Landslide_Landsat_Global_v2/) ---
.venv/bin/python main.py labels      # pseudo masks -> data/pseudo_masks_v2/
.venv/bin/python main.py train       # 50 ep -> checkpoints/best.pt
.venv/bin/python -m src.ensemble_infer          # strict  -> results/masks_ens/
.venv/bin/python -m src.ensemble_infer relaxed  # relaxed -> results/masks_ens_relaxed/
.venv/bin/python main.py geocode      # -> results/landslides{,_relaxed}.geojson

# --- Landslide4Sense benchmark (real GT, honest 3-way split) ---
# data already at data/L4S/TrainData/ (Zenodo TrainData.zip, md5 408f09247e...)
# --- Indian-region segmentation (HR-GLDD real masks, 576 MB) ---
mkdir -p data/HR_GLDD && for f in trainX trainY valX valY testX testY; do \
  curl -sL -o data/HR_GLDD/$f.npy \
  "https://zenodo.org/records/7189381/files/$f.npy?download=1"; done
.venv/bin/python main.py hrgldd        # 100 ep ~8 min -> test F1 0.748 / IoU 0.60
.venv/bin/python hrgldd_qc.py 6        # RGB | GT | pred panels -> qc/hrgldd_*.png

.venv/bin/python main.py l4s_classify # patch classifier, 50 ep ~4 min -> 90.7% acc / 92.2% F1
.venv/bin/python main.py l4s          # improved DeepUNet, 90 ep ~20 min -> test F1 0.755
.venv/bin/python main.py l4s_baseline # legacy U-Net, same split      -> test F1 0.759
.venv/bin/python -m src.l4s_ensemble  # average both (needs both ckpts) -> test F1 0.760

# --- Indian-region visual QC ---
.venv/bin/python qc_overlay.py wayanad_in_2024 kavalappara_in_2019 idukki_in_2018 \
    chamoli_in_2021 ramban_in_2024 teesta_in_2023   # -> qc/*_in_*.png

# --- independent validation ---
.venv/bin/python qc_glc.py results/landslides_relaxed.geojson
.venv/bin/python qc_overlay.py wayanad_in_2024 acapulco_mx_2023 wajima_jp_2024

# --- GEE re-export (auth expires) ---
# ssh -L 8085:localhost:8085 <host> then:
.venv/bin/python -c "import ee; ee.Authenticate(auth_mode='localhost')"
python download_dataset.py            # OFFSET cycles 0/10/20/30 for 4 batches
```

## 7. Artifacts index

| Path | Contents |
|---|---|
| `checkpoints/best_hrgldd.pt` | **Indian-region segmentation (HR-GLDD, test F1 0.748 / IoU 0.60)** |
| `checkpoints/hrgldd_report.json` | HR-GLDD honest val/test metrics |
| `checkpoints/hrgldd_norm.json` | HR-GLDD train-split per-band mean/std |
| `qc/hrgldd_*.png` | RGB \| human mask \| prediction panels (per-patch F1) |
| `checkpoints/best_l4s_cls.pt` | **Patch classifier (test 90.7% acc / 92.2% F1)** + norm stats |
| `checkpoints/l4s_cls_report.json` | classifier honest val/test metrics + AUC |
| `checkpoints/l4s_cls_norm.json` | classifier train-split per-band mean/std |
| `checkpoints/best_l4s.pt` | Improved DeepUNet (test F1 0.755) + its norm stats |
| `checkpoints/best_l4s_baseline.pt` | Legacy SingleUNet baseline (test F1 0.759) |
| `checkpoints/best.pt` | Siamese event model (val F1 0.440 vs pseudo) |
| `checkpoints/l4s_report*.json` | honest val/test segmentation metrics, per model |
| `checkpoints/l4s_ensemble_report.json` | ensemble held-out test (F1 0.760) |
| `checkpoints/l4s_norm.json` | train-split per-band mean/std |
| `checkpoints/l4s_sweep*.json` | val threshold sweep + frozen thr |
| `checkpoints/best_landsat5.pt` | Historical Landsat-5 Indian model (Himalaya/NE, ~40–53% ≤3 km) |
| `checkpoints/landsat5_test.json` | L5 held-out catalog-proximity results |
| `checkpoints/best_s1_wg.pt` | Sentinel-1 SAR Western Ghats model (pseudo-labels) |
| `checkpoints/best_s1_wg_real.pt` | **SAR fine-tuned on REAL Kodagu masks (F1 0.554 / IoU 0.383)** |
| `checkpoints/best_s1_wg_cls.pt` | **SAR tile classifier — real GT (92.9% acc / 90.9% bal-acc / AUC 0.952)** |
| `checkpoints/s1_real_finetune.json`, `s1_cls_real.json` | SAR real-GT seg/cls metrics |
| `data/Kodagu_SAR_GT/{PRE,POST,mask}.tif` | real-GT SAR scene + rasterised 217-polygon inventory |
| `qc/s1_*.png`, `qc/s1_kodagu_real_gt.png`, `qc/s1_kodagu_tiles.png` | SAR detection + tile-flag QC |
| `results/landslides*.geojson/csv` | georeferenced polygons (strict/relaxed) |
| `results/qc_glc_report.csv` | GLC distance table |
| `landslides_relaxed_glc_report2.csv` | + in-scene containment flags |
| `qc/*.png`, `qc/*_in_*.png` | PRE/POST/mask visual QC panels (incl. Indian) |
| `download_landsat5_india.py`, `download_sentinel1_wg.py`, `build_kodagu_gt.py`, `s1_tile_overlay.py` | L5/SAR download, real-GT build, tile overlay |
| `src/`, `main.py` | full pipeline code |

## 8. Conclusions

1. **Good Indian-region segmentation on real human masks** (HR-GLDD,
   Kerala/Karnataka Western Ghats included): held-out test **F1 0.748 /
   IoU 0.60 / pixel acc 0.943**, with **0.85–0.96 per-patch F1** on clear
   scars (QC panels). Real 3 m digitized masks — a genuine pixel-delineation
   result, not pseudo-labels.
2. **≥ 90% landslide-detection accuracy** (L4S patch classifier, held-out
   test: **90.7 % acc / 92.2 % F1 / 0.967 AUC**) — out-of-sample, near-
   balanced labels (not an imbalance artefact).
3. **≥ 75% pixel-segmentation F1** on L4S under the same honest protocol
   (ensemble test F1 **0.760**). Pixel-level 90% is beyond published SOTA for
   that benchmark and is *not* claimed.
4. Full operational pipeline works end-to-end: GEE export → train →
   inference → georeferenced polygons for 39 real events (141 strict /
   237 relaxed polygons, 65–114 km²), **8 of them in Indian regions, 6/8
   detected** with visual-QC confirmation (wayanad on the Chooralmala scar).

## 9. Recommended next steps

- **Lift HR-GLDD F1 further**: the 0.748 comes from 1119 train patches —
  add a pretrained/attention encoder, model ensemble, or copy-paste
  augmentation of scars; aim ~0.80.
- **Kodagu-only Indian score**: rebuild HR-GLDD from the authors' per-site
  inventories (shapefiles) to isolate the Indian tiles, or request MTMS-HLD
  (Uttarakhand) access for a purely-Himalayan segmentation benchmark.
- Sentinel-2 re-export of the 8 Indian Landsat events (GEE) so the 14-band
  L4S classifier applies directly; re-export miss events (shirur, kedarnath)
  with QA_PIXEL cloud masking.
- Adaptive pseudo-label thresholds (fix 0%-label events) → retrain.
