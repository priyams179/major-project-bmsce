# UAV Flood Mapping & Asset Impact Assessment (FloodNet + Indian Flood Regions)

An end-to-end deep learning framework for aerial drone (UAV) disaster response. The system performs two-stage analysis:
1. **Asset Detection & Bounding Box Localization**: Identifies and classifies physical assets (**Buildings** and **Vehicles/Cars**) before flood mapping.
2. **Flood Semantic Segmentation**: Maps pixel-level inundation boundaries (Water, Flooded Roads, Flooded Structures) using DeepLabV3+ with ASPP and Focal-Dice loss.
3. **Spatial Impact Reasoning**: Fuses bounding box footprints with the flood inundation mask to compute exact structural submergence percentages and classify infrastructure into disaster risk tiers.
4. **Indian Flood Domain Adaptation**: Specialized transforms simulating alluvial sediment/muddy floodwaters (Brahmaputra, Kosi, Ganga plains), diverse roofing materials (Mangalore terracotta tiles, corrugated tin, concrete terraces), and monsoon atmospheric conditions.

---

## Directory Structure

```
flood-management-uav/
├── config.yaml               # Central project and hyperparameter configuration
├── requirements.txt          # Python dependencies
├── data/
│   ├── download_floodnet.py  # Downloader and realistic benchmark chip generator
│   ├── prepare_detection_data.py # Converts FloodNet masks to YOLO bounding box format
│   └── dataset.py            # PyTorch Dataset supporting multi-class & Indian augmentations
├── models/
│   ├── detector.py           # YOLOv8 aerial detector wrapper & tiled inference
│   ├── segmentor.py          # DeepLabV3+ with ResNet backbone and FocalDiceLoss
│   └── domain_adaptation.py   # Turbidity, roofing diversity, and monsoon transforms
├── train_detector.py         # Script to fine-tune YOLO asset detector
├── train_segmentor.py        # Script to train DeepLabV3+ flood segmentor
├── pipeline.py               # End-to-end inference: Image -> Detection -> Flood Map -> Overlays
├── visualize.py              # Visual engine (HUD Scoreboard, composite overlays, side-by-side)
├── evaluate.py               # Comprehensive mIoU and mAP benchmark suite
└── outputs/                  # Generated flood maps, bounding boxes, and JSON disaster reports
```

---

## Environment Setup (uv Managed)

All packages and environments are managed with `uv`:

```bash
# Navigate to the project directory
cd flood-management-uav

# Create virtual environment with Python 3.12
uv venv .venv --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Quickstart: Step-by-Step Workflow

### 1. Prepare Data & Annotations

Generate benchmark UAV flood chips and extract YOLO bounding boxes for buildings and cars:

```bash
# Generate benchmark chips (or download FloodNet dataset)
uv run python data/download_floodnet.py --samples 24

# Extract YOLO bounding boxes from segmentation masks
uv run python data/prepare_detection_data.py
```

### 2. Train the Models

#### A. Train Asset Detector (Buildings & Vehicles)
```bash
uv run python train_detector.py --epochs 10 --batch-size 8
```

#### B. Train Flood Semantic Segmentor (DeepLabV3+)
```bash
uv run python train_segmentor.py --epochs 15 --batch-size 4
```

### 3. Run End-to-End Inference & Impact Assessment

Run the complete pipeline on any UAV aerial image:

```bash
uv run python pipeline.py --image data/floodnet/images/uav_flood_0001.jpg --output-dir outputs
```

**Generated Outputs**:
- `outputs/<name>_flood_mapping_result.png`: Composite drone photo overlaid with translucent flood inundation mask, color-coded bounding boxes (Red = Submerged, Orange = Threatened, Green = Safe, Cyan = Submerged Car), and an executive HUD disaster scoreboard.
- `outputs/<name>_flood_mask.png`: Binary inundation mask (1=Flooded, 0=Dry).
- `outputs/<name>_disaster_report.json`: Quantitative disaster metrics including total flooded percentage, submerged building count, and NDRF/SDMA severity category.

### 4. Evaluate Performance

Compute dataset-wide mIoU, class-specific IoUs (Water, Flooded Roads, Flooded Buildings), and detector mAP:

```bash
uv run python evaluate.py
```

---

## Indian Flood Domain Adaptation Details

Standard models trained on Western flood datasets underperform in Indian flood scenarios due to three major domain shifts:
1. **Turbid / Muddy Alluvial Water**: In Assam (Brahmaputra) and Bihar (Kosi/Ganga), floodwaters carry high sediment loads, appearing opaque brown or clay-yellow rather than clear or blue-green. The `TurbidMuddyWaterTransform` dynamically shifts water hue, saturation, and luminance to ensure robust detection of silt-laden waters.
2. **Roof Material Heterogeneity**: Indian settlements feature high textural contrast between terracotta Mangalore tiles, corrugated tin/GI sheets, and flat RCC roofs. The `IndianRoofMaterialAugment` injects texture and spectral variations to prevent false negatives.
3. **Monsoon Atmospheric Conditions**: Diffuse light and humid haze are simulated via `MonsoonAtmosphereAugment` to maintain high recall during overcast monsoon drone sorties.

