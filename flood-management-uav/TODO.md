# FloodNet Implementation TODO

## Agenda

Reproduce the FloodNet semi-supervised classification and segmentation work from:

- Repository: https://github.com/sahilkhose/FloodNet
- Paper: https://arxiv.org/abs/2105.08655
- Dataset: FloodNet Track 1 (`FloodNet-Supervised_v1.0` style layout)

Target: a clean Python package managed with `uv`, with command-line entry points for data download/preparation, training, evaluation, prediction, and a Streamlit demo.

Primary deliverables:

1. ResNet18 semi-supervised binary flood classification.
2. DeepLabV3+ with EfficientNet-B3 semi-supervised 10-class flood segmentation.
3. Notebook-compatible preprocessing, pseudo-label schedules, metrics, and inference.
4. Reproducible `uv` environment and tests.
5. README and commands that a future agent can run without guessing.

## Locked Decisions

- Use `uv` for virtual environment and dependency management.
- Use Python 3.12.
- Use HuggingFace dataset mirror by default; support manual Dropbox zip as fallback.
- Implement best models reported in the paper:
  - Classification: ResNet18.
  - Segmentation: DeepLabV3+ with EfficientNet-B3 encoder.
- Use full paper schedules:
  - Classification: 200 epochs.
  - Segmentation: 100 supervised epochs + 100 semi-supervised epochs.
- Match the reference notebooks' behavior, including legacy quirks. Do not silently "fix" pseudo-label behavior, output inversion, or class mapping without documenting it.
- Add a fixed random seed for reproducibility. The notebooks were not fully deterministic; document this deviation.
- Keep data, checkpoints, and outputs out of git.

## Current Workspace State

- `flood-management-uav/` currently contains only `.gitignore`.
- Previous tracked YOLO/UAV pipeline files are deleted in the worktree. Do not restore them unless explicitly requested.
- This TODO describes a fresh implementation.

## Source Facts To Preserve

### Dataset layout

```text
floodnet/
├── Train/
│   ├── Labeled/
│   │   ├── Flooded/
│   │   │   ├── image/          # 51 images
│   │   │   └── mask/           # 51 masks, *_lab.png
│   │   └── Non-Flooded/
│   │       ├── image/          # 347 images
│   │       └── mask/           # 347 masks, *_lab.png
│   └── Unlabeled/
│       └── image/              # 1047 images
├── Validation/
│   └── image/                  # 450 images, no masks
├── Test/
│   └── image/                  # 448 images, no masks
└── class_mapping.csv
```

- Images are `4000x3000` (predominantly) and `4592x3072` JPEGs.
- Masks are indexed grayscale PNGs with values `0..9`, sized `4000x3000`.
- Some masks are cropped relative to their images (or images are wider). Resize masks nearest-neighbor to align with image dimensions before preprocessing and caching.
- Official Validation/Test splits have no labels. They are used as unlabeled data for pseudo-labeling.
- Local evaluation must use labeled splits derived from the 398 training labels.

### Class IDs

```text
0 Background
1 Building Flooded
2 Building Non-Flooded
3 Road Flooded
4 Road Non-Flooded
5 Water
6 Tree
7 Vehicle
8 Pool
9 Grass
```

Binary flood classes: `[1, 3, 5]`.

### Reference notebook behavior: classification (`FloodNet_T1.ipynb`)

- Images resized with `cv2.resize(..., (400, 300))`, giving width 400, height 300.
- Labeled data split with `train_test_split(train_size=0.5, stratify=y)`.
  - 199 train / 199 validation from the 398 labeled images.
- Unlabeled pool includes:
  - `Train/Unlabeled/image` (1047)
  - `Validation/image` (450)
  - `Test/image` (448)
  - Total: 1945 images, sorted.
- Transforms:
  - `RandomHorizontalFlip`
  - `RandomVerticalFlip`
  - `ToTensor`
  - Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- Sampler: `WeightedRandomSampler` with inverse class frequency, replacement enabled.
- Model: torchvision `resnet18(pretrained=True)`, final `Linear(512, 2)`.
- Loss: `CrossEntropyLoss`.
- Optimizer: `SGD(lr=0.01)`.
- Batch size: 64.
- Epochs: 200.
- Alpha schedule:
  - `epoch < 10`: alpha = 0
  - `10 <= epoch < 150`: `numpy.linspace(0, 1, 140)[epoch - 10]`
  - `epoch >= 150`: alpha = 1
- Pseudo labels:
  - Regenerate after each epoch where `epoch >= 9`.
  - Run on the shuffled unlabeled loader.
  - Store `argmax(logits)` by unlabeled dataset index.
  - Unlabeled loss: `alpha * CrossEntropyLoss(logits, pseudo_label_by_index)`.
  - Skip unlabeled phase when alpha is 0.
- Validation: macro F1; save best checkpoint.
- Test submission output:
  - Notebook writes `image_classes.json`.
  - It inverts predictions with `pred ^ 1` because classification labels were opposite of submission labels.
  - Preserve this behind `legacy_output_inversion: true`.

### Reference notebook behavior: segmentation (`FloodNet_T2.ipynb`)

- Images and masks resized to `512x512`.
- Image preprocessing:
  - `cv2.bilateralFilter(image, 5, 75, 75)`
  - `cv2.dilate(..., 2x2 kernel, iterations=2)`
  - `cv2.erode(..., 2x2 kernel, iterations=1)`
- Labeled split: `train_test_split(test_size=0.3, shuffle=True, random_state=SEED)`.
  - 278 train / 120 validation from 398 labeled images.
- Notebook uses training augmentation for validation too. Preserve this.
- Train transforms:
  - `A.Resize(512, 512)`
  - `A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5)`
  - `A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5)`
  - `A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
  - To tensor.
- Masks: `cv2.imread(mask)[:, :, 0]`, then one-hot with 10 classes.
- Model:

```python
segmentation_models_pytorch.DeepLabV3Plus(
    encoder_name="efficientnet-b3",
    encoder_depth=5,
    decoder_channels=256,
    classes=10,
)
```

- Encoder weights: ImageNet.
- Loss: `BCEWithLogitsLoss` + Dice, equal weight (`bce_weight=0.5`, `smooth=1e-5`).
- Supervised stage:
  - Optimizer: Adam, `lr=1e-3`.
  - Batch size: 8.
  - Epochs: 100.
- Semi-supervised stage:
  - Resume supervised checkpoint.
  - Optimizer: SGD, `lr=0.01`.
  - Batch size: 8.
  - Epochs: 100.
  - Unlabeled set: `Train/Unlabeled/image[0:-500]` -> 547 images.
  - Alpha schedule:
    - `epoch < 15`: alpha = 0
    - `epoch >= 15`: `numpy.linspace(0, 0.5, 655 - 15)[epoch - 15]`
    - If index exceeds array: use last value.
  - Consequence: with only 100 epochs, max alpha is about `0.0657`. Preserve.
- Pseudo labels:
  - Generate after each epoch where `epoch >= 14`.
  - Store raw model logits per batch, not argmax masks.
  - Unlabeled loss:

```python
alpha * (
    BCEWithLogits(pred, pseudo_logits)
    + Dice(sigmoid(pred), pseudo_logits)
)
```

  - Use pseudo-label batch index, matching the notebook.
  - This is legacy behavior. README must warn that paper mIoU may not reproduce exactly with this loop.
- Metrics:
  - mIoU = mean of per-class IoU over flattened argmax predictions.
  - Also track per-class IoU, pixel accuracy, confusion matrix.
  - Save best checkpoint by validation mIoU.
  - Save checkpoints every 5 epochs.
- Inference:
  - Resize input to `512x512`.
  - Apply bilateral filter, dilate 2 iterations, erode 1 iteration.
  - Normalize and tensor.
  - Forward, sigmoid, argmax.
  - Resize mask to `4000x3000`.
  - Erode with `3x3` kernel once.
  - Save indexed PNG.

## Target Project Structure

```text
flood-management-uav/
├── pyproject.toml
├── uv.lock
├── README.md
├── TODO.md
├── configs/
│   ├── base.yaml
│   ├── classification.yaml
│   ├── segmentation.yaml
│   └── smoke.yaml
├── src/floodnet/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── constants.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── prepare.py
│   │   ├── classification_dataset.py
│   │   ├── segmentation_dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── segmentor.py
│   ├── losses.py
│   ├── metrics.py
│   ├── semi_supervised.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   ├── train_segmentor.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── visualization.py
├── demo/
│   └── app.py
├── tests/
│   ├── test_alpha_schedule.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_data.py
│   └── test_smoke.py
├── checkpoints/              # gitignored
├── outputs/                  # gitignored
└── data/                     # gitignored
```

## Implementation Phases

### Phase 0: Scaffold

- [x] Create `pyproject.toml` with package metadata and dependencies.
- [x] Configure PyTorch CUDA wheel index (`cu124` preferred; verify availability).
- [x] Add `ruff` and `pytest` configuration.
- [x] Add `src/floodnet/__init__.py`, `cli.py`, `config.py`, `constants.py`.
- [x] Add YAML configs with paper defaults.
- [x] Add CLI subcommands:
  - `download`
  - `prepare`
  - `train-classifier`
  - `train-segmentor`
  - `evaluate`
  - `predict`
- [x] Update `.gitignore` for `data/`, `checkpoints/`, `outputs/`, `*.pt`, `*.pth`, `runs/`, `.venv/`.

### Phase 1: Data Download And Validation

- [x] `download.py`: HuggingFace mirror `torchgeo/floodnet`, file `floodnet.zip`, resume-capable.
- [x] Support `--zip` for manually downloaded official Dropbox archive.
- [x] Verify file SHA256 when known.
- [x] Safe extraction; reject path traversal.
- [x] `validate.py`:
  - [x] Verify expected counts: `51`, `347`, `1047`, `450`, `448`.
  - [x] Verify image dimensions are `3000x4000` / `4592x3072`.
  - [x] Verify mask values are within `0..9`.
  - [x] Handle cropped masks via nearest resize.
  - [x] Report missing image/mask pairs.
- [x] `prepare.py`:
  - [x] Build deterministic manifests for each task and split.
  - [x] Classification cache: resize to width 400, height 300.
  - [x] Segmentation cache: resize to 512x512; apply bilateral + dilate/erode to images; nearest resize masks.
  - [x] Classification labeled split: 50/50 stratified.
  - [x] Segmentation labeled split: 70/30 with fixed seed.
  - [x] Classification unlabeled pool: train unlabeled + validation + test.
  - [x] Segmentation unlabeled pool: first 547 of train unlabeled.
  - [x] Save manifests as JSON with paths, labels, split, source.
  - [x] Add `--dry-run`.

### Phase 2: Shared Core

- [x] `constants.py`: class names, class colors, flood classes, ImageNet stats.
- [x] `losses.py`:
  - [x] Classification cross-entropy.
  - [x] Segmentation BCE + Dice with equal weights.
  - [x] Legacy pseudo-logit loss path for segmentation.
- [x] `metrics.py`:
  - [x] Accuracy and macro F1.
  - [x] Per-class precision/recall/F1.
  - [x] Confusion matrix.
  - [x] mIoU, per-class IoU, pixel accuracy, frequency-weighted IoU.
- [x] `semi_supervised.py`:
  - [x] Classification alpha scheduler exact values.
  - [x] Segmentation alpha scheduler exact values.
  - [x] Pseudo-label containers matching notebook indexing.
- [x] Checkpoint utilities: save/load model, optimizer, epoch, config, metrics.
- [x] Seeding and deterministic dataloader worker setup.

### Phase 3: Classification Pipeline

- [x] `classification_dataset.py`:
  - [x] Load cached 400x300 images.
  - [x] Apply flips, tensor conversion, ImageNet normalization.
  - [x] Return image, label, index, path.
- [x] `classifier.py`: ResNet18 with ImageNet weights and 2-logit head.
- [x] `train_classifier.py`:
  - [x] Weighted sampler.
  - [x] SGD `lr=0.01`, batch 64.
  - [x] 200 epochs.
  - [x] Alpha ramp from epoch 10 to 150.
  - [x] Regenerate pseudo labels after epoch 9.
  - [x] Unlabeled loss with dataset-indexed pseudo labels.
  - [x] Validate with macro F1.
  - [x] Save `best.pt`, `last.pt`, CSV/TensorBoard metrics.
- [x] `evaluate.py` classification mode:
  - [x] Accuracy, F1, per-class metrics, confusion matrix.
  - [x] JSON output.
- [x] Classification test prediction:
  - [x] Write `image_classes.json`.
  - [x] Apply legacy `pred ^ 1` inversion when configured.

### Phase 4: Segmentation Pipeline

- [x] `segmentation_dataset.py`:
  - [x] Load 512x512 cached images/masks.
  - [x] Apply notebook train transform to train and validation.
  - [x] Return image, one-hot or class mask, path.
  - [x] Unlabeled dataset returns image only.
- [x] `segmentor.py`: SMP DeepLabV3Plus EfficientNet-B3, 10 classes, ImageNet encoder.
- [x] `train_segmentor.py` supervised stage:
  - [x] Adam `lr=1e-3`, batch 8, 100 epochs.
  - [x] BCE + Dice loss.
  - [x] Validation mIoU.
  - [x] Save supervised checkpoint.
- [x] `train_segmentor.py` semi-supervised stage:
  - [x] Resume supervised checkpoint.
  - [x] SGD `lr=0.01`, batch 8, 100 epochs.
  - [x] Alpha schedule start 15, end 655, max 0.5.
  - [x] Generate raw-logit pseudo labels after epoch 14.
  - [x] Legacy batch-index pseudo-label loss.
  - [x] Save every 5 epochs and best by mIoU.
- [x] `evaluate.py` segmentation mode:
  - [x] mIoU, per-class IoU, pixel accuracy, confusion matrix.
  - [x] Save colorized predictions and overlays.
  - [x] JSON output.
- [x] `predict.py`:
  - [x] Single image and folder support.
  - [x] 512 resize, notebook preprocessing.
  - [x] Sigmoid + argmax, resize to original dimensions or 4000x3000 per config.
  - [x] 3x3 erosion.
  - [x] Save indexed mask, color mask, overlay, JSON summary.
  - [x] Direct notebook-compatible resizing pipeline.

### Phase 5: Demo And Visualization

- [x] `visualization.py`:
  - [x] Class color palette.
  - [x] Mask colorization.
  - [x] Alpha overlay.
  - [x] Legend.
  - [x] Side-by-side image/mask view.
- [x] `demo/app.py` Streamlit:
  - [x] Upload image.
  - [x] Task selector.
  - [x] Show input, predicted mask, overlay, legend.
  - [x] Download result.
  - [x] Use checkpoint paths from config.

### Phase 6: Tests And Verification

- [x] Unit tests for alpha scheduler exact values.
- [x] Unit tests for BCE + Dice loss.
- [x] Unit tests for mIoU/F1 known arrays.
- [x] Unit tests for manifest pairing and mask decoding.
- [x] Unit tests for classification pseudo-label index mapping.
- [x] Unit tests for legacy segmentation pseudo-logit loss.
- [x] Synthetic 64x64 dataset smoke test:
  - [x] 1 classification epoch on CPU.
  - [x] 1 segmentation epoch on CPU.
- [x] Run `uv run pytest -q`.
- [x] Run `ruff check .`.
- [x] Run `uv run floodnet prepare --dry-run`.
- [x] Real-data gate:
  - [x] Validate dataset counts.
  - [x] Run 1 GPU epoch for classification.
  - [x] Run 1 GPU epoch for supervised segmentation.
  - [x] Run 1 GPU epoch for semi-supervised segmentation.
- [x] Full runs ready and documented for full schedules.

## Environment And Commands

```bash
uv python pin 3.12
uv venv --python 3.12
uv sync
```

Planned dependencies:

```text
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python-headless
numpy
pillow
scikit-learn
pyyaml
tqdm
tensorboard
matplotlib
huggingface-hub
streamlit
pytest
ruff
```

Planned commands:

```bash
uv run floodnet download --dest data/raw
uv run floodnet prepare --config configs/base.yaml

uv run floodnet train-classifier --config configs/classification.yaml
uv run floodnet train-segmentor --config configs/segmentation.yaml --stage supervised
uv run floodnet train-segmentor --config configs/segmentation.yaml --stage semi-supervised --resume checkpoints/seg_supervised_best.pt

uv run floodnet evaluate --task classification --split validation --checkpoint checkpoints/cls_best.pt
uv run floodnet evaluate --task segmentation --split validation --checkpoint checkpoints/seg_best.pt

uv run floodnet predict --task segmentation --input path/to/image.jpg --output outputs
uv run streamlit run demo/app.py
```

## Acceptance Criteria

- `uv sync` succeeds with Python 3.12 and CUDA-enabled PyTorch.
- `floodnet download` retrieves and extracts the dataset, or clearly instructs how to use `--zip`.
- `floodnet prepare` validates all dataset counts and creates manifests/caches.
- `pytest` and `ruff` pass.
- Synthetic smoke tests complete on CPU.
- Classification and segmentation training can run for 1 epoch on GPU without errors.
- Checkpoints resume correctly.
- Evaluation produces metrics JSON and colorized predictions.
- README documents:
  - Environment setup.
  - Dataset download.
  - Full training commands.
  - Evaluation and inference.
  - Legacy notebook quirks and deviations.
  - Paper-reference metrics and why local numbers may differ.

## Known Risks And Warnings

- Dataset zip is about 12.8 GB. Extracted data plus caches needs substantial disk space.
- Full schedules are long; use `configs/smoke.yaml` for tests but keep paper schedules as defaults.
- Official Validation/Test splits have no ground truth. Local metrics use notebook-style labeled splits.
- Legacy segmentation pseudo-label loop uses raw logits and batch indexing. This may not reproduce the paper mIoU. Preserve by request, document clearly.
- Legacy classification output inversion (`pred ^ 1`) is required for notebook-compatible submission output. Keep it explicit and tested.
- Some masks are cropped relative to images. Always resize masks nearest to image dimensions before applying joint transforms.
- Do not commit credentials, dataset archives, checkpoints, or generated outputs.

## Definition Of Done

- [x] All phases above complete.
- [x] Full training commands documented and verified for at least 1 epoch.
- [x] Tests and lint pass.
- [x] README and TODO reflect final implementation.
- [x] No dataset, checkpoints, or outputs committed.
