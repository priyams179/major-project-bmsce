# AGENTS.md

Guide for AI agents working in the `flood-management-uav` codebase.

## Project Overview

End-to-end deep learning system for UAV (drone) flood disaster response. Two-stage pipeline:
1. **YOLOv8** detects assets (Buildings, Vehicles) with bounding boxes.
2. **DeepLabV3+** (ResNet-50 backbone) performs 10-class semantic segmentation of flood scenes.
3. Spatial overlay fuses the two to compute per-asset submergence ratios and an NDRF/SDMA severity tier.

Includes Indian flood domain adaptation (turbid alluvial water, heterogeneous roofing, monsoon haze) via Albumentations transforms.

## Environment & Commands

This project uses **`uv`** (not raw pip/conda). Python 3.12.

```bash
# Setup (one-time)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

The `--extra-index-url` for PyTorch CUDA 12.1 wheels is **required** — plain `uv pip install -r requirements.txt` will fail or pull CPU-only wheels.

```bash
# Generate synthetic FloodNet benchmark chips + masks
uv run python data/download_floodnet.py --samples 24

# Convert FloodNet masks -> YOLO bounding box dataset
uv run python data/prepare_detection_data.py

# Train
uv run python train_detector.py --epochs 10 --batch-size 8
uv run python train_segmentor.py --epochs 15 --batch-size 4

# End-to-end inference on a single image
uv run python pipeline.py --image data/floodnet/images/uav_flood_0001.jpg --output-dir outputs

# Evaluate (writes evaluation_results.json)
uv run python evaluate.py
```

All scripts accept `--config` (default `config.yaml`) and most accept `--epochs`, `--batch-size`, `--lr` overrides. Run any script with `--help` for full flags.

There is **no test suite, no linter, no CI, no Makefile**. Verification is done by running `evaluate.py` or `pipeline.py`.

## Architecture & Data Flow

```
config.yaml  (single source of truth: paths, hyperparams, class defs, thresholds)
     |
     +-- data/download_floodnet.py   -> data/floodnet/{images,masks}/  (synthetic generator; auto-invoked by trainers if data missing)
     +-- data/prepare_detection_data.py -> data/yolo_detection/        (mask contours -> YOLO txt + floodnet_detect.yaml)
     +-- data/dataset.py               -> FloodNetSegmentationDataset   (PyTorch Dataset; applies domain adaptation pipeline)
     +-- models/detector.py            -> UAVAssetDetector (YOLO wrapper, standard + tiled inference)
     +-- models/segmentor.py           -> DeepLabV3Plus, FocalDiceLoss, UAVFloodSegmentor (inference wrapper)
     +-- models/domain_adaptation.py   -> Albumentations pipeline with Indian flood transforms
     +-- train_detector.py             -> fine-tunes YOLO via ultralytics API
     +-- train_segmentor.py            -> trains DeepLabV3+ with AMP + CosineAnnealingLR; exports compute_iou/update_confusion_matrix (reused by evaluate.py)
     +-- pipeline.py                   -> UAVFloodMappingPipeline: detection -> segmentation -> spatial overlay -> JSON report + visualization
     +-- visualize.py                  -> generate_composite_flood_map (HUD overlay), generate_side_by_side_comparison (matplotlib 3-panel)
     +-- evaluate.py                   -> mIoU + mAP benchmark; imports compute_iou/update_confusion_matrix FROM train_segmentor
```

**Pipeline control flow** (`pipeline.py` `process_image`):
1. Load image (path or RGB array).
2. If image > 1200 px in either dimension, run **tiled inference** (`predict_tiled`) with global per-class NMS; else standard `predict`.
3. Run segmentor `predict` -> returns `pred_mask` (10-class), `flood_mask` (binary), `flood_prob` (probability map).
4. For each detected box, crop `flood_mask`, compute submergence ratio, classify impact using thresholds from `config.yaml` under `impact_assessment`.
5. Compute scene-level severity tier (Level 1/2/3) based on flood % and submerged-building ratio.
6. Write `_disaster_report.json`, `_flood_mask.png`, and `_flood_mapping_result.png` (composite HUD overlay via `visualize.py`).

## Class ID Conventions (Critical)

There are **two different class ID spaces** — do not confuse them:

| Segmentation (10 classes, `config.yaml` `classes.segmentation`) | Detection / YOLO (3 classes, `config.yaml` `classes.detection`) |
|---|---|
| 0=Background, 1=Building Flooded, 2=Building Non-Flooded, 3=Road Flooded, 4=Road Non-Flooded, 5=Water, 6=Tree, 7=Vehicle, 8=Pool, 9=Grass | 0=Building-Flooded, 1=Building-Non-Flooded, 2=Vehicle |

**Flood classes** (binary flood mask = 1): segmentation IDs **[1, 3, 5]** (Building Flooded, Road Flooded, Water). Defined as `FLOOD_CLASSES` in both `data/dataset.py` and `models/segmentor.py`.

**Mask-to-YOLO mapping** (`FLOODNET_TO_YOLO` in `data/prepare_detection_data.py`): `{1:0, 2:1, 7:2}` — only buildings and vehicles are extracted; roads/water/trees are ignored for detection.

## Gotchas & Non-Obvious Patterns

- **Weight loading is CUDA-gated** (`models/segmentor.py` `UAVFloodSegmentor.__init__`): checkpoint is only loaded `if weights_path is not None and torch.cuda.is_available()`. On CPU, weights are silently ignored even if a valid path is provided. If running on CPU, the segmentor uses random-initialized weights.

- **Tiled inference threshold**: `pipeline.py` triggers `predict_tiled` only when `w > 1200 or h > 1200`. This 1200 px threshold is hardcoded, not in `config.yaml`. Tiling uses `tile_size`/`overlap` from config (default 1024 / 0.20).

- **Mask path resolution** (`data/dataset.py` `_find_mask_path`): looks for `base.png`, `base_lab.png`, `base_mask.png`, `base.jpg` in that order. FloodNet convention is `*_lab.png`. If no mask found, `__getitem__` returns a dict without `"mask"` key (inference-only mode) — training code assumes mask is always present.

- **Auto data generation**: Both `train_detector.py` and `train_segmentor.py` auto-invoke `create_verification_samples` (synthetic data) if `data/floodnet/images/` is empty. The synthetic generator (`download_floodnet.py`) alternates Indian-turbid and coastal-clear scenes every other sample.

- **YOLO dataset auto-prep**: `train_detector.py` calls `prepare_yolo_dataset` automatically if `floodnet_detect.yaml` is missing. Images are **symlinked** (not copied) into `data/yolo_detection/images/{train,val}/` by default — pass `--copy` to `prepare_detection_data.py` to copy instead.

- **Cross-module imports**: `evaluate.py` imports `compute_iou` and `update_confusion_matrix` from `train_segmentor` (not from a shared utils module). If you refactor metrics, update both call sites. `data/dataset.py` has a try/except fallback import for `domain_adaptation` to handle being run as a package vs. script.

- **Lazy import in pipeline**: `visualize.py`'s `generate_composite_flood_map` is imported inside `process_image` (not at module top) to avoid matplotlib dependency at pipeline construction time.

- **Albumentations 2.0+ API**: `get_indian_flood_augmentation_pipeline` uses `A.RandomResizedCrop(size=(h, w), ...)` and `A.Resize(height=h, width=w)` — the `size=`/`height=`+`width=` keyword form, not the legacy `height=`/`width=` on `RandomResizedCrop`. Requirements pin `albumentations>=1.4.0` but the code targets 2.0+ API.

- **Normalization**: ImageNet stats `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` are used in three places independently (dataset transform, `segmentor.py` `predict` preprocessing, `domain_adaptation.py` val transform). They are not centralized — keep them in sync if changing.

- **Color space**: `visualize.py` and `models/detector.py` use **BGR** color tuples for OpenCV drawing. `config.yaml` `colors` section uses **RGB**. `pipeline.py` internally works in RGB (converts `cv2.imread` BGR->RGB at entry). Watch the color space when crossing module boundaries.

- **Checkpoint paths**: Segmentor saves to `checkpoints/best_segmentor.pth` and `checkpoints/latest_segmentor.pth`. YOLO saves to `checkpoints/yolo_detector/weights/best.pt`. Both `checkpoints/` and `outputs/` and `data/floodnet/` and `data/yolo_detection/` are **gitignored** — they will not exist on a fresh clone.

- **Evaluation reads saved results**: `evaluation_results.json` is committed and reflects a prior run (mIoU 0.26, mAP50 0.31) on synthetic data. Re-running `evaluate.py` overwrites it.

## Config-Driven Design

`config.yaml` is the single source of truth. Key sections:
- `project.device`: `"cuda"` — falls back to CPU at runtime if CUDA unavailable.
- `paths`: `data_root` (FloodNet), `yolo_data_root`, `models_dir` (`checkpoints`), `output_dir` (`outputs`).
- `classes.segmentation` / `classes.detection`: class ID -> name maps (different ID spaces, see above).
- `detector` / `segmentor`: hyperparameters; trainers read from here unless CLI-overridden.
- `domain_adaptation`: toggle/probability for turbidity, roof, monsoon augmentations.
- `impact_assessment`: submergence thresholds (building 15%, threatened 5%, vehicle 10%).

## Naming & Style Conventions

- Python files are flat in root (`pipeline.py`, `evaluate.py`, `train_*.py`, `visualize.py`) with two packages: `models/` and `data/`.
- Classes use PascalCase; functions use snake_case; constants use UPPER_SNAKE (`FLOOD_CLASSES`, `CLASS_NAMES`).
- Inference wrappers follow the `UAV<Name>` pattern: `UAVAssetDetector`, `UAVFloodSegmentor`, `UAVFloodMappingPipeline`.
- Detection results are always a list of dicts with keys `class_id`, `class_name`, `box` (xyxy), `confidence`.
- Docstrings are Google-style with Args/Returns sections on public methods.
- Scripts use `argparse` with `--config` defaulting to `config.yaml`.
