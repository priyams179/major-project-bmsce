# Sentinel-1 Deep Learning Flood Detection (Sen1Floods11 & U-Net)

A deep learning and remote sensing pipeline for high-precision flood detection from Sentinel-1 Synthetic Aperture Radar (SAR) imagery, trained on the global **Sen1Floods11** benchmark dataset.

---

## Overview

1. **Benchmark Ground Truth (Sen1Floods11)**:
   - 446 hand-labeled $512 \times 512$ chips across 11 global flood events in 6 continents.
   - Dual-polarization Sentinel-1 SAR imagery (VV and VH channels).
   - Validated binary ground truth with cloud/nodata pixel masking (`-1` ignore label).

2. **Deep Learning Architecture (U-Net)**:
   - 4-level encoder-decoder U-Net with skip connections and batch normalization.
   - Dual-channel SAR input (`[VV, VH]`) or 3-channel input (`[VV, VH, VH - VV]`).
   - Custom **Masked BCE + Dice Loss** and **Masked Focal-Tversky Loss** to handle class imbalance while ignoring cloud/nodata regions.

3. **Exact Evaluation Metrics**:
   - Global Confusion Matrix tracker computing true dataset-level **IoU (Jaccard Index)**, **F1 / Dice Score**, **Precision**, **Recall**, and **Accuracy**.

---

## Setup

```bash
# Clone and setup environment with uv
uv venv .venv --python 3.12

# Install dependencies
uv pip install torch rasterio numpy scipy matplotlib
```

---

## Quickstart

### 1. Download Sen1Floods11 Dataset
Download the official hand-labeled SAR chips and split CSVs directly from Google Cloud Storage:
```bash
uv run python sen1floods11.py --download --workers 16
```

### 2. Train the U-Net Model
Train the U-Net on the official Sen1Floods11 train split (252 chips) with real-time validation (89 chips) and final evaluation on the test set (90 chips):
```bash
uv run python train_cnn.py \
    --epochs 30 \
    --batch-size 16 \
    --lr 1e-3 \
    --in-channels 2 \
    --loss bce_dice \
    --output-dir models
```

**Training outputs**:
- `models/best_model.pth`: Checkpoint with highest validation IoU.
- `models/final_model.pth`: Final trained weights.
- `models/training_curves.png`: Training & validation loss and IoU curves.
- `models/training_history.json`: Epoch-by-epoch metric logs.

### 3. Run Inference & Visualization
Run inference on any test chip or external Sentinel-1 GeoTIFF:
```bash
# Evaluate on default test chip
uv run python predict_cnn.py --model models/best_model.pth

# Evaluate on a specific chip ID
uv run python predict_cnn.py --model models/best_model.pth --chip-id Boliva_103757

# A full Sentinel-1 scene (2-band VV/VH GeoTIFF), optionally cropped to a lon/lat box
uv run python predict_cnn.py --model best_model.pth --input-s1 scene.tif --bbox 92.6 26.4 93.1 26.8
```

Scenes of any size are processed in 2048 px blocks, each predicted in overlapping
512 px tiles (the training chip size), so memory stays at a few hundred MB.
Pixels with no radar data are written as 255 (excluded) - they normalise to
-35 dB, which the model would otherwise call water. `final_model.pth` (bare
weights) loads as well as `best_model.pth`.

The resulting `*_disaster_map.tif` is the input to `../Clustering/sar_scenes.py
ingest`, which turns it into priority zones and evacuation routes.

> **Normalisation fix.** `normalize_s1_band` used to treat a band as linear power
> whenever any pixel exceeded 1, so dB chips with one bright pixel (67 of the 68
> Assam chips) were log-transformed twice and mostly blanked. It now uses the sign
> (linear power is never negative). The current weights were trained with the old
> behaviour, yet score better with the fix (Assam IoU 0.40 -> 0.52); retraining
> with it should help further.

**Inference outputs**:
- `predictions/<chip_id>_flood_pred.tif`: Binary GeoTIFF flood map.
- `predictions/<chip_id>_flood_prob.tif`: Continuous probability GeoTIFF.
- `predictions/<chip_id>_comparison.png`: 4-panel visual comparison (VV, VH, Ground Truth, U-Net Prediction).

---

## Project Structure

```
├── sen1floods11.py      # Sen1Floods11 downloader and PyTorch Dataset class
├── cnn_model.py         # U-Net architecture, Masked Losses, and Metric Tracker
├── train_cnn.py         # Training loop, validation, and test benchmarking
├── predict_cnn.py       # Inference and side-by-side comparison generator
├── geocode.py           # Geocoding utility for raw Sentinel-1 GRD products
├── flood_mapping.py     # Heuristic baseline SAR flood mapping (Refined Lee)
├── data/
│   └── sen1floods11/    # Hand-labeled S1 chips, labels, and split CSVs
├── models/              # Checkpoints, training logs, and loss curves
└── predictions/         # Output GeoTIFFs and visualization PNGs
```
