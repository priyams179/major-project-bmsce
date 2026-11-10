# Landslide Management — Landsat Collection-2 Pipeline

Pre/post Landsat 8/9 detection → georeferenced landslide polygons, plus
real-mask landslide segmentation on Indian-region high-resolution data
(HR-GLDD, incl. Kerala/Karnataka Western Ghats), with honest held-out
benchmarking against human labels (HR-GLDD + Landslide4Sense).

## Quick start

```bash
uv sync

.venv/bin/python main.py labels        # pseudo-labels (multi-index consensus)
.venv/bin/python main.py train         # Siamese U-Net, 50 epochs
.venv/bin/python -m src.ensemble_infer # strict ensemble masks
.venv/bin/python main.py geocode       # GeoJSON + CSV polygons
# Indian-region segmentation (HR-GLDD real masks; download 576 MB -> data/HR_GLDD/)
for f in trainX trainY valX valY testX testY; do \
  curl -sL -o data/HR_GLDD/$f.npy "https://zenodo.org/records/7189381/files/$f.npy?download=1"; done
.venv/bin/python main.py hrgldd        # DeepUNet+NDVI seg -> test F1 0.748 / IoU 0.60
.venv/bin/python hrgldd_qc.py 6        # RGB | GT | prediction panels -> qc/hrgldd_*.png

# Other benchmarks
.venv/bin/python main.py l4s_classify  # patch classifier -> 90.7% acc / 92.2% F1 (real GT)
.venv/bin/python main.py l4s           # L4S segmentation, improved model (honest test F1 0.755)
.venv/bin/python -m src.l4s_ensemble   # average seg models -> test F1 0.760
.venv/bin/python qc_overlay.py wayanad_in_2024 kavalappara_in_2019  # Indian Landsat QC

# Historical Landsat-5 Indian model (2007-2012 events; needs Earth Engine auth)
.venv/bin/python download_landsat5_india.py --n 150  # cloud-filtered L5 pre/post -> data/Landsat5_India/
.venv/bin/python main.py l5_labels   # change-detection pseudo-labels
.venv/bin/python main.py l5_train    # SiameseUNet (3-band L5) -> best_landsat5.pt
.venv/bin/python main.py l5_test     # held-out detection rate vs catalog points

# Sentinel-1 SAR Western Ghats model (cloud-penetrating; needs Earth Engine auth)
.venv/bin/python download_sentinel1_wg.py --n 40  # S1 GRD VV,VH,slope pre/post -> data/Sentinel1_WG/
.venv/bin/python main.py s1_labels   # SAR change-detection pseudo-labels (ΔVV/ΔVH on slopes)
.venv/bin/python main.py s1_train    # SiameseUNet (6-ch SAR) -> best_s1_wg.pt
.venv/bin/python main.py s1_test     # held-out detection rate (Western Ghats)
# REAL ground truth: Kodagu-2018 inventory (217 polygons, CC BY 4.0)
.venv/bin/python build_kodagu_gt.py       # S1 over inventory extent + rasterised real mask
.venv/bin/python main.py s1_eval_real     # score pseudo-trained model vs real GT (F1 0.28)
.venv/bin/python main.py s1_train_real    # fine-tune on real masks -> held-out REAL F1 0.55
.venv/bin/python main.py s1_cls_real      # tile-level classifier -> 92.9% acc / 90.9% bal-acc
.venv/bin/python s1_tile_overlay.py       # highlight flagged tiles vs truth -> qc/s1_kodagu_tiles.png
```

## Results at a glance

| Check | Number |
|---|---|
| **Western Ghats SAR — tile detection, REAL ground truth** (Kodagu 2018, held-out block) | **92.9% accuracy · 90.9% balanced-acc · AUC 0.952** (F1 82.2%) — "does this tile contain a landslide?" |
| **Western Ghats SAR — pixel segmentation, REAL ground truth** (same) | **F1 0.554 · IoU 0.383** (vs 0.284 pseudo-trained, **+95%**) — exact scar outlines; SAR penetrates monsoon cloud where Landsat-5 scored 0% |
| Sentinel-1 SAR (catalog-proximity, 30 events) | 57% ≤3 km · 4/4 hits ≤2 km — detects Wayanad, Kodagu, Kavalappara, Taliye |
| **Historical Landsat-5 Indian model** (Himalaya/NE, held-out, catalog-proximity) | **~40–53% ≤3 km · ~56–63% ≤5 km** (±split variance). Western Ghats = 0% (monsoon cloud) → use the SAR / Sentinel-2 models there |
| **Indian-region segmentation (HR-GLDD, real high-res masks, held-out test)** | **F1 0.748 · IoU 0.60 · pixel acc 0.943** — incl. Kerala/Karnataka Western Ghats (Kodagu) |
| **Landslide-presence accuracy (patch classifier, held-out test, real labels)** | **90.7% accuracy · 92.2% F1 · 0.967 AUC** |
| L4S segmentation F1 (held-out test, honest) | 0.760 ensemble / 0.759 baseline / 0.755 improved (Sentinel-2, 10 m) |
| Protocol | author or 3-way split; threshold frozen on val; test touched once |
| Indian Landsat events | 8 events (Kerala, Uttarakhand, J&K, Karnataka, Sikkim); 6/8 detected; QC in `qc/*_in_*.png` |
| Polygons mapped (strict / relaxed) | 141 / 237, 65.0 / 113.5 km² |

## Run on your own scene (download / upload → classify + segment)

Feed a real satellite scene through the trained classifier + segmentation
models to get georeferenced landslide outputs.

```bash
# (optional) download NASA Earthdata HLS (Sentinel-2) for an AOI + a DEM
.venv/bin/python download_earthdata.py --hls --bbox 76.0 11.3 76.3 11.6 \
    --start 2024-07-25 --end 2024-08-10 --out data/scenes
.venv/bin/python download_earthdata.py --nasadem --bbox 76.0 11.3 76.3 11.6 --out data/dem

# run on a Sentinel-2 .SAFE (uploaded from Copernicus CDSE), HLS folder, or GeoTIFF
.venv/bin/python run_scene.py --safe data/scenes/S2B_..._.SAFE \
    --dem data/dem/dem.tif --bbox 76.0 11.3 76.3 11.6 --out results/scene_aoi
.venv/bin/python run_scene.py --hls data/scenes/HLS.S30.T43PFS.2024213 --out results/hls
.venv/bin/python run_scene.py --geotiff my_14band_stack.tif --out results/custom
```

**Outputs** (in `--out`): `seg_prob.tif`, `seg_mask.tif`, `landslides.geojson`
(polygons + area_km²), `tiles.csv` (per-tile classification), `overlay.png`,
`summary.json`. Add `--gate` to keep segmentation only inside tiles the
classifier flags as landslide.

**Bands the models need (14, Sentinel-2 L4S order):** B01, B02, B03, B04, B05,
B06, B07, B08, B09, B10, B11, B12, **slope**, **DEM**. Notes: B10 exists only
in L1C `.SAFE` (L2A → zero-filled); B8A is *not* used; slope/DEM come from a
DEM you supply (`--dem`; e.g. NASADEM) else they are zero-filled. All bands are
resampled to a common 10 m grid. Inputs are **scene-standardised** by default
(`--norm scene`), which is robust when a scene's reflectance scale differs from
L4S. Sensor note: `.SAFE` = ESA/Copernicus (10 m); NASA Earthdata serves **HLS**
(30 m) which the pipeline also ingests — both carry the required optical bands.
See [src/scene.py](src/scene.py) for the full band table.

> Caveat: the models are trained on Landslide4Sense; running on arbitrary real
> scenes is a domain-shift scenario. Scene-standardisation mitigates it, but for
> operational accuracy fine-tune on labelled local data.

## Read this first

**[REPORT.md](REPORT.md)** — full technical report: data, method, results,
failure analysis, reproduction commands, artifact index.
