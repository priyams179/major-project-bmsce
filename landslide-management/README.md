# Multi-Modal Landslide Detection & Mapping Pipeline (India Focus)

An end-to-end, production-grade deep learning and remote sensing pipeline for **landslide detection and mapping**, tailored for India's high-vulnerability terrain: the **Western Ghats**, the **Himalayas**, and **Northeast India**.

The system ingests **Sentinel-1 SAR**, **Landsat-8/9 Collection-2 L2** optical, and **Copernicus DEM (GLO-30)** elevation data to train semantic segmentation and bi-temporal change detection models, generating geocoded disaster response GeoTIFFs (EPSG:4326).

The active stack is 24 channels in a fixed order: 7 Sentinel-1
(`vv_pre, vh_pre, vv_post, vh_post, delta_vv, delta_vh, vh_vv_ratio`), 9 Landsat
(`SR_B2…SR_B7, NDVI, NDWI, BSI`), 8 DEM features. Sentinel-2 and the
Landslide4Sense benchmark remain supported as a legacy path, not the default:
L4S labels are derived from Sentinel-2 imagery, so keeping it as the primary
training set would evaluate the model on the sensor it is meant to replace.

---

## 1. Key Highlights & India Monsoon-Aware Design

- **All-Weather Monsoon Operation (June – September):**
  During the Indian Southwest Monsoon, persistent cloud cover reaches **85%–98%** over the Western Ghats and Himalayas. Optical satellites are blinded for **14 to 85+ days**. This pipeline prioritizes **Sentinel-1 C-band SAR (VV + VH)** to achieve rapid detection within **24–48 hours** of an event.
- **Dual-Model Strategy:**
  - **Phase 1: Multi-Modal Semantic Segmentation (`ResNet34UNet` with ASPP):**
    Pretrained ImageNet ResNet-34 backbone adapted for arbitrary input channels (7-channel SAR-only monsoon mode, 16-channel S1+Landsat, or the full 24-channel S1+Landsat+DEM stack) with Atrous Spatial Pyramid Pooling for multi-scale feature extraction. The channel count is read from the dataset manifest rather than passed by hand.
  - **Phase 2: Bi-Temporal Change Detection (`SNUNetCD`):**
    Siamese ResNet-18 encoders with dense nested skip connections (U-Net++ topology) and Ensemble Channel Attention Modules (ECAM) to compare pre- and post-disaster passes and isolate newly triggered scars from permanent bare ground.
- **Extreme Class Imbalance Mitigation:**
  Landslides occupy $<1\%$ of scene pixels. The pipeline utilizes **Masked Focal-Tversky Loss** ($\alpha=0.3, \beta=0.7, \gamma=1.33$) to heavily penalize false negatives and maximize recall on scarce landslide bodies while cleanly ignoring cloud/nodata pixels.
- **Operational Disaster Response Post-Processing:**
  - Horn's algorithm slope filtering: excludes flat terrain ($<5^\circ$) where landslides cannot initiate and vertical cliffs ($>70^\circ$).
  - Pre-existing bare rock separation using historical NDVI/BSI.
  - 8-connectivity connected component filtering to suppress radar speckle noise.
  - Native radar tie-point and UTM-to-WGS84 (EPSG:4326) geocoding.

---

## 2. Directory Structure

```
landslide-management/
├── requirements.txt             # Python dependencies
├── README.md                    # System documentation and operational guide
│
├── download_landsat.py          # Landsat-8/9 Collection-2 L2 search & download via USGS M2M
├── download_sentinel.py         # Sentinel-1 (CDSE) & Copernicus DEM downloader with India event profiles
├── download_l4s.py              # Landslide4Sense benchmark downloader (legacy, Sentinel-2 based)
│
├── preprocess_sar.py            # Sentinel-1 GRD calibration (DN -> σ° dB), noise removal, Refined Lee,
│                                #   phase-correlation co-registration, 7-band bitemporal stack
├── preprocess_optical.py        # Landsat/Sentinel-2 BOA reflectance, QA cloud masking, NDVI/NDWI/BSI, dNDVI
├── preprocess_dem.py            # Copernicus DEM GLO-30 terrain features: Slope, Aspect, Curvatures, TWI, TRI
├── prepare_dataset.py           # Inventory polygon rasterisation, multi-modal stacking, event-level CV splits
│
├── models/                      # Deep learning architecture package
│   ├── __init__.py              # Public model and loss exports
│   ├── segmentation_model.py    # ResNet-34 U-Net + ASPP & standard U-Net with N-channel transfer
│   ├── change_detection_model.py# SNUNet-CD Siamese ResNet-18 with nested skips and ECAM attention
│   ├── losses.py                # Masked Focal-Tversky, Masked BCE+Dice, Combined Loss
│   └── metrics.py               # ConfusionMatrixTracker (IoU, F1, Prec, Rec) & optimal threshold search
│
├── train_segmentation.py        # Phase 1 training loop with Cosine Annealing, mixed precision (AMP)
├── train_change_detection.py    # Phase 2 bi-temporal change detection training script
│
├── predict.py                   # Single-chip inference and multi-panel inspection
├── predict_fullscene.py         # Sliding-window inference with 2D Hann tapering and 4-fold TTA
├── postprocess.py               # Operational slope range filtering, small cluster cleanup, multi-class map
├── geocode.py                   # Radar geometry and UTM to WGS84 (EPSG:4326) geocoding
│
├── evaluate.py                  # Evaluation report, Precision-Recall curve, confusion matrix
└── visualize.py                 # Multi-panel comparison plots, full-scene color overlays, training curves
```

---

## 3. Pre-Configured Indian Landslide Events

| Event Identifier | Date | Coordinates | Sentinel-2 Tile | Sentinel-1 Track | DEM Tile | Fatalities / Impact |
|:---|:---|:---|:---|:---|:---|:---|
| `wayanad_2024` | 2024-07-30 | 11.51°N, 76.15°E | `43PGP` | Track 63 | `N11_E076` | >400 deaths, Chooralmala-Mundakkai debris flow |
| `chamoli_2021` | 2021-02-07 | 30.37°N, 79.73°E | `44RMR` | Track 27 | `N30_E079` | 204 deaths, Ronti Peak rock-ice avalanche |
| `joshimath_2023`| 2023-01-01 | 30.55°N, 79.56°E | `44RMR` | Track 129 | `N30_E079` | Regional town subsidence & slope creep |
| `irshalwadi_2023`| 2023-07-19 | 18.93°N, 73.23°E | `43QCB` | Track 63 | `N18_E073` | 86 deaths, steep basaltic scarp failure |
| `tupul_2022`    | 2022-06-29 | 24.71°N, 93.64°E | `46RFS` | Track 143 | `N24_E093` | 61 deaths, railway yard cut-slope collapse |
| `pettimudi_2020`| 2020-08-06 | 10.17°N, 77.02°E | `43PEP` | Track 63 | `N10_E077` | 70 deaths, Rajamala tea estate debris slide |

---

## 4. Installation & Environment Setup

Using `uv` (recommended) or standard Python `venv`:

```bash
cd landslide-management

# Using the existing project environment
source ../flood-management/.venv/bin/activate

# Or install in a fresh virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. End-to-End Operational Workflow

### Step 1: Download & Inspect Data
```bash
# List all pre-configured Indian landslide events
python download_sentinel.py --list-events

# Sentinel-1 GRD pair (pre/post), keyless via the sentinel-s1-l1c mirror;
# with CDSE_CLIENT_ID/SECRET set it uses the official CDSE zip route instead
python download_sentinel.py --event wayanad_2024 --sar --output data/raw/s1

# Landsat-8/9 Collection-2 L2, keyless by default (Planetary Computer mirror);
# set USGS_M2M_API_KEY (or pass --backend m2m) for the official USGS route
python download_landsat.py --event wayanad_2024 --output data/raw/landsat
python download_landsat.py --list-events
python download_landsat.py --event wayanad_2024 --search-only   # coverage check

# Copernicus DEM GLO-30
python download_sentinel.py --event wayanad_2024 --dem-only --output data/raw/dem
```

No credentials are required: both downloaders fall back to anonymous open-data
mirrors (Microsoft Planetary Computer for Landsat, Element84's `sentinel-s1-l1c`
bucket for Sentinel-1) and write the same directory layouts the preprocess steps
expect. Optional credentials switch to the official agency routes:

| Variable | Used by | Source | Without it |
|---|---|---|---|
| `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET` | Sentinel-1 (official CDSE zips) | [dataspace.copernicus.eu](https://dataspace.copernicus.eu) | keyless `sentinel-s1-l1c` mirror |
| `USGS_M2M_API_KEY` | Landsat-8/9 (official USGS M2M) | [m2m.cr.usgs.gov](https://m2m.cr.usgs.gov) | keyless Planetary Computer mirror (`--backend pc`) |

If you do want the official USGS route, generating the key is only half of it.
M2M access has to be approved separately: in [ERS](https://ers.cr.usgs.gov) open
**Access Request → Access Controls → Request Access**, choose *Access to EE's
Machine to Machine interface (MACHINE)*, and wait for the USGS team to grant it.
Requests are refused with a permission error until then, which looks like a bad
key.

`download_landsat.py` keeps only the `SR_B*` bands plus `QA_PIXEL`, on both
backends. That is deliberate: the M2M bundle product returns every file of a
scene, thermal bands and browse included, at roughly an order of magnitude more
bytes for two events' worth of data we never read.

### Step 2: Preprocess Radar, Optical, and Topographic Data
```bash
# 1. SAR: calibration to σ° dB, noise removal, Refined Lee speckle filter,
#    phase-correlation co-registration, then the 7-band bitemporal stack
python preprocess_sar.py --pre data/raw/s1/S1_pre.SAFE --post data/raw/s1/S1_post.SAFE \
    --output output/sar/wayanad_2024
# -> output/sar/wayanad_2024/change/sar_stack.tif  (vv_pre … vh_vv_ratio)

# 2. Optical: Landsat SR rescaling, QA cloud masking, NDVI, NDWI, BSI
python preprocess_optical.py --pre data/raw/landsat/L8_pre --post data/raw/landsat/L8_post \
    --sensor landsat --output output/optical/wayanad_2024

# 3. DEM features: slope (Horn), aspect, curvatures, TWI, TRI
python preprocess_dem.py --aoi wayanad_2024 \
    --target-raster output/sar/wayanad_2024/change/sar_stack.tif --output output/dem
```

### Step 3: Prepare Dataset & Event-Level Splits
```bash
# Multi-event dataset from inventory polygons (active pipeline, 24 channels)
python prepare_dataset.py events \
    --config data/config/events.json \
    --output data/processed/s1_landsat \
    --patch-size 256 --stride 128 \
    --val-events tupul_2022 pettimudi_2020 \
    --test-events irshalwadi_2023 joshimath_2023

# Single scene
python prepare_dataset.py custom --config scene_config.json --output data/processed/wayanad

# Legacy Landslide4Sense benchmark (Sentinel-2, 14 channels)
python prepare_dataset.py l4s --l4s-dir data/l4s --output data/processed/l4s
```

`events.json` gives one entry per event, each with either a `mask` GeoTIFF or a
`polygon_file`, plus the raster paths:

```json
{
  "events": [
    {
      "event_name": "wayanad_2024",
      "lat": 11.51, "lon": 76.15,
      "polygon_file": "data/inventory/wayanad_2024.geojson",
      "sar_stack": "output/sar/wayanad_2024/change/sar_stack.tif",
      "landsat":   "output/optical/wayanad_2024/post/stack.tif",
      "dem":       "output/dem/wayanad_2024/terrain_stack.tif"
    }
  ],
  "val_events":  ["tupul_2022", "pettimudi_2020"],
  "test_events": ["irshalwadi_2023", "joshimath_2023"]
}
```

Channel order is fixed and written to `dataset_meta.json`: 7 Sentinel-1, then
9 Landsat, then 8 DEM. Splitting is by whole event, so the test score measures
transfer to terrain the model never saw — adjacent patches of one landslide
share slope, land cover and radar geometry, and splitting them randomly inflates
the score.

Pixels outside every inventory polygon are marked invalid and skipped by the
loss. A configurable ring around each polygon is kept as real negatives
(`--negative-margin`, default 150 m), because that band is where a fresh scar
must be told apart from the stable slope above it.

### Step 4: Model Training

#### Phase 1: Semantic Segmentation (ResNet-34 U-Net + ASPP)
```bash
python train_segmentation.py \
    --arch resnet34_unet \
    --data-dir data/processed/s1_landsat \
    --epochs 50 \
    --batch-size 16 \
    --loss focal_tversky \
    --mixed-precision \
    --output-dir models/segmentation
```

`--in-channels` is read from the dataset manifest and should not be passed. The
manifest is authoritative; a flag that disagrees with it is reported and
overridden.

#### Phase 2: Bi-Temporal Change Detection (SNUNet-CD)
```bash
python train_change_detection.py \
    --arch snunet \
    --in-channels 7 \
    --data-dir data/processed/bitemporal \
    --epochs 50 \
    --batch-size 8 \
    --output-dir models/change_detection
```

### Step 5: Evaluation & Quality Metrics
```bash
# Compute exact global dataset metrics and generate PR curve
python evaluate.py \
    --model models/segmentation/best_model.pth \
    --data-dir data/processed/s1_landsat \
    --split test \
    --pr-curve \
    --output-dir output/eval
```

**Reference targets (Landslide4Sense test split, Sentinel-2 baseline):**
- **IoU:** $\ge 55.0\%$ (SOTA baseline: $\sim 53\% - 58\%$)
- **F1-Score / Dice:** $\ge 70.0\%$ (SOTA baseline: $\sim 73\% - 78\%$)
- **Recall:** $\ge 75.0\%$ (Prioritized via $\beta=0.7$ Focal-Tversky weighting)
- **Precision:** $\ge 65.0\%$

These numbers come from the L4S benchmark and are not directly comparable to
the six-event S1+Landsat split. With one event per split, held-out IoU varies far
more between events than it does between random patch subsets: Irshalwadi and
Joshimath differ in terrain, season and radar geometry. Expect the spread across
events to be the honest signal, and treat any single figure with suspicion.

### Step 6: Full-Scene Operational Prediction & Post-Processing
```bash
# 1. Full-scene sliding-window inference with 2D Hann blending and 4-fold TTA
python predict_fullscene.py \
    --model models/segmentation/best_model.pth \
    --input data/processed/scenes/wayanad_2024_stack.tif \
    --output output/wayanad_prob.tif \
    --patch-size 512 \
    --stride 256

# 2. Operational slope range filtering (5° - 70°) and small-cluster cleanup
python postprocess.py \
    --prob-map output/wayanad_prob.tif \
    --slope output/dem/slope_deg.tif \
    --threshold 0.50 \
    --min-slope 5.0 \
    --max-slope 70.0 \
    --min-size 10 \
    --output output/wayanad_landslide_response_map.tif

# 3. Geocode to standard EPSG:4326 (ready for QGIS / Bhuvan overlay)
python geocode.py \
    --input output/wayanad_landslide_response_map.tif \
    --reproject \
    --output output/wayanad_landslide_map_wgs84.tif
```

### Step 7: Visualization Suite
```bash
# Visualize single-patch input, prediction, and ground truth
python visualize.py --type patch --image data/processed/l4s/test/image_00001.npy --output figures/patch_001.png

# Single-chip inference with an RGB composite chosen by band name
python predict.py \
    --model models/segmentation/best_model.pth \
    --input data/processed/s1_landsat/test/test_image_00000.npy \
    --mask data/processed/s1_landsat/test/test_mask_00000.npy \
    --composite-from-checkpoint \
    --composite false_color \
    --output output/predictions

# Plot training and validation loss/IoU curves
python visualize.py --type training --history models/segmentation/training_history.json --output figures/training_curves.png

# Render full-scene georeferenced disaster map with legend
python visualize.py --type fullscene --map output/wayanad_landslide_response_map.tif --output figures/wayanad_map.png
```

---

## 6. Multi-Class Map Output Schema

The final post-processed GeoTIFF contains 4 standardized operational classes:

| Class Value | Class Meaning | Map Color | Description |
|:---:|:---|:---|:---|
| **0** | **Stable Terrain** | Green | Unaffected vegetated slopes, stable valleys, plains |
| **1** | **Newly Triggered Landslide** | Crimson Red | Active scar, crown detachment, or debris flow track |
| **2** | **Pre-Existing Bare Rock/Soil** | Dark Orange | Historical rock outcrops, active stone quarries, gravel riverbeds |
| **255** | **Excluded Terrain** | Gray / NoData | Gentle slopes ($<5^\circ$), sheer cliffs ($>70^\circ$), cloud/sensor nodata |

