# Multi-Modal Landslide Detection & Mapping Pipeline (India Focus)

An end-to-end, production-grade deep learning and remote sensing pipeline for **landslide detection and mapping**, tailored for India's high-vulnerability terrain: the **Western Ghats**, the **Himalayas**, and **Northeast India**.

The system ingests **Sentinel-1 SAR**, **Sentinel-2 MSI / Landsat Optical**, and **Copernicus DEM (GLO-30)** elevation data to train semantic segmentation and bi-temporal change detection models, generating geocoded disaster response GeoTIFFs (EPSG:4326).

---

## 1. Key Highlights & India Monsoon-Aware Design

- **All-Weather Monsoon Operation (June – September):**
  During the Indian Southwest Monsoon, persistent cloud cover reaches **85%–98%** over the Western Ghats and Himalayas. Optical satellites are blinded for **14 to 85+ days**. This pipeline prioritizes **Sentinel-1 C-band SAR (VV + VH)** to achieve rapid detection within **24–48 hours** of an event.
- **Dual-Model Strategy:**
  - **Phase 1: Multi-Modal Semantic Segmentation (`ResNet34UNet` with ASPP):**
    Pretrained ImageNet ResNet-34 backbone adapted for arbitrary input channels (6-channel SAR-only monsoon mode or 14-channel full multi-modal mode) with Atrous Spatial Pyramid Pooling for multi-scale feature extraction.
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
├── download_l4s.py              # Landslide4Sense benchmark dataset downloader & validator
├── download_sentinel.py         # Sentinel-1/2 & Copernicus DEM downloader with India event profiles
│
├── preprocess_sar.py            # Sentinel-1 GRD calibration (DN -> σ° dB), noise removal, Refined Lee
├── preprocess_optical.py        # Sentinel-2/Landsat BOA reflectance, SCL cloud masking, NDVI/dNDVI/BSI
├── preprocess_dem.py            # Copernicus DEM GLO-30 terrain features: Slope, Aspect, Curvatures, TWI, TRI
├── prepare_dataset.py           # Co-registration, multi-modal stacking, balanced patch extraction, spatial CV
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

# Download Copernicus DEM GLO-30 for Wayanad
python download_sentinel.py --event wayanad_2024 --dem-only --output data/raw/dem

# Check or organize Landslide4Sense benchmark dataset
python download_l4s.py --check --data-dir data/l4s
```

### Step 2: Preprocess Radar, Optical, and Topographic Data
```bash
# 1. SAR Preprocessing: Calibration to σ° dB, noise removal, Refined Lee speckle filter
python preprocess_sar.py --pre data/raw/S1_pre.SAFE --post data/raw/S1_post.SAFE --output output/sar

# 2. Optical Preprocessing: SCL cloud masking, reflectance rescaling, NDVI, dNDVI, BSI
python preprocess_optical.py --pre data/raw/S2_pre --post data/raw/S2_post --sensor sentinel2 --output output/optical

# 3. DEM Topographic Feature Extraction: Slope (Horn's), Aspect, Curvatures, TWI, TRI
python preprocess_dem.py --aoi wayanad --target-raster output/sar/vv_norm.tif --output output/dem
```

### Step 3: Prepare Dataset & Spatial Cross-Validation Splits
```bash
# Process Landslide4Sense benchmark (128x128 patches, 14 channels)
python prepare_dataset.py l4s --l4s-dir data/l4s --output data/processed/l4s

# Or process custom Indian multi-modal disaster stack
python prepare_dataset.py custom --config scene_config.json --output data/processed/wayanad
```

### Step 4: Model Training

#### Phase 1: Semantic Segmentation (ResNet-34 U-Net + ASPP)
```bash
python train_segmentation.py \
    --arch resnet34_unet \
    --in-channels 14 \
    --data-dir data/processed/l4s \
    --epochs 50 \
    --batch-size 16 \
    --loss focal_tversky \
    --mixed-precision \
    --output-dir models/segmentation
```

#### Phase 2: Bi-Temporal Change Detection (SNUNet-CD)
```bash
python train_change_detection.py \
    --arch snunet \
    --in-channels 5 \
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
    --data-dir data/processed/l4s \
    --split test \
    --pr-curve \
    --output-dir output/eval
```

**Target Benchmarks (Landslide4Sense Test Split):**
- **IoU:** $\ge 55.0\%$ (SOTA baseline: $\sim 53\% - 58\%$)
- **F1-Score / Dice:** $\ge 70.0\%$ (SOTA baseline: $\sim 73\% - 78\%$)
- **Recall:** $\ge 75.0\%$ (Prioritized via $\beta=0.7$ Focal-Tversky weighting)
- **Precision:** $\ge 65.0\%$

### Step 6: Full-Scene Operational Prediction & Post-Processing
```bash
# 1. Full-scene sliding-window inference with 2D Hann blending and 4-fold TTA
python predict_fullscene.py \
    --model models/segmentation/best_model.pth \
    --input data/raw/wayanad_stacked_scene.tif \
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

