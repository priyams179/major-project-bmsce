# FloodNet UAV Flood Classification And Segmentation

Reproduction of the FloodNet semi-supervised classification and segmentation
work from [sahilkhose/FloodNet](https://github.com/sahilkhose/FloodNet) and
[arXiv:2105.08655](https://arxiv.org/abs/2105.08655).

The project is managed with `uv`, uses Python 3.12, and exposes a single
`floodnet` command for download, preparation, training, evaluation, and
inference.

## Models

| Task | Model | Checkpoint |
| --- | --- | --- |
| Binary flood classification | ResNet18, 2 logits | `checkpoints/cls_best.pt` |
| 10-class segmentation | DeepLabV3+ / EfficientNet-B3 | `checkpoints/seg_best.pt` |

Default schedules are the notebook schedules:

- Classification: 200 epochs, SGD `lr=0.01`, batch 64.
- Segmentation: 100 supervised epochs, then 100 semi-supervised epochs,
  batch 8.

Paper reference results:

- Classification: `96.70%` test accuracy, `98.10%` F1.
- Segmentation: `0.5223` mIoU with pseudo-labels for DeepLabV3+.

Local numbers can differ because the official Validation and Test splits have
no masks. Local validation uses a seeded split of the 398 labeled training
images.

## Setup

```bash
uv python pin 3.12
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`uv sync` installs a CUDA-enabled PyTorch wheel on Linux. If CUDA is unavailable,
check the installed wheel and install a matching CUDA build if needed.

## Dataset

The HuggingFace mirror is the default source:

```bash
uv run floodnet download --dest data/raw
```

This downloads `torchgeo/floodnet` / `floodnet.zip` (about 12.8 GB), verifies
SHA256 `962e6dac95495d7fdc249681d3973889f7e93487fb9e3da25e1df4f451f4f272`,
and extracts the expected layout under `data/raw/floodnet`.

If automatic download fails, download the official Dropbox archive and pass it
locally:

```bash
uv run floodnet download --dest data/raw --zip /path/to/floodnet.zip --no-verify
```

Expected layout:

```text
data/raw/floodnet/
├── Train/
│   ├── Labeled/
│   │   ├── Flooded/{image,mask}          # 51 image/mask pairs
│   │   └── Non-Flooded/{image,mask}      # 347 image/mask pairs
│   └── Unlabeled/image/                  # 1047 images
├── Validation/image/                     # 450 images, unlabeled
└── Test/image/                           # 448 images, unlabeled
```

Build validated manifests and resized caches:

```bash
uv run floodnet prepare --config configs/base.yaml
```

`prepare` validates counts, image/mask pairing, and mask values. Classification
images are cached at width 400, height 300. Segmentation images are cached at
512x512 after the notebook's bilateral/dilate/erode preprocessing; masks are
resized with nearest-neighbor interpolation.

Useful options:

```bash
uv run floodnet prepare --dry-run
uv run floodnet prepare --limit 20 --force
```

## Training

Classification:

```bash
uv run floodnet train-classifier --config configs/classification.yaml
```

Segmentation:

```bash
uv run floodnet train-segmentor --config configs/segmentation.yaml --stage both
```

Split the segmentation stages:

```bash
uv run floodnet train-segmentor --config configs/segmentation.yaml --stage supervised
uv run floodnet train-segmentor --config configs/segmentation.yaml \
  --stage semi-supervised --resume checkpoints/seg_supervised_best.pt
```

Smoke runs use one epoch, small batches, and no data augmentation mismatch:

```bash
uv run floodnet train-classifier --config configs/classification.yaml --smoke --device cpu --no-pretrained
uv run floodnet train-segmentor --config configs/segmentation.yaml --stage both --smoke --device cpu --no-pretrained
```

Training writes:

- `checkpoints/cls_best.pt`, `checkpoints/cls_last.pt`
- `checkpoints/seg_supervised_best.pt`, `checkpoints/seg_supervised_last.pt`
- `checkpoints/seg_best.pt`, `checkpoints/seg_last.pt`
- CSV metrics under `checkpoints/`
- TensorBoard logs under `outputs/tensorboard/`
- JSON summaries under `outputs/`

## Evaluation

Classification validation:

```bash
uv run floodnet evaluate --task classification --split val \
  --checkpoint checkpoints/cls_best.pt
```

Segmentation validation:

```bash
uv run floodnet evaluate --task segmentation --split val \
  --checkpoint checkpoints/seg_best.pt
```

The evaluation JSON includes aggregate metrics, per-class results, confusion
matrices, and classification predictions. Segmentation evaluation also writes
indexed masks, colorized masks, and overlays under
`outputs/segmentation/eval_<split>/predictions/`.

Official Validation/Test images have no ground truth. For classification
inference on those images:

```bash
uv run floodnet evaluate --task classification --split official_test \
  --checkpoint checkpoints/cls_best.pt --legacy-invert
```

## Prediction

Classification on an image or folder:

```bash
uv run floodnet predict --task classification --input path/to/image.jpg \
  --output outputs/classification --checkpoint checkpoints/cls_best.pt
```

The notebook applies `pred ^ 1` to submission labels. That behavior is enabled
by default through `legacy.classification_output_inversion: true`. Disable it
with `--no-legacy-invert` when you want the model's raw class:

```bash
uv run floodnet predict --task classification --input path/to/image.jpg \
  --output outputs/classification --checkpoint checkpoints/cls_best.pt \
  --no-legacy-invert
```

Segmentation:

```bash
uv run floodnet predict --task segmentation --input path/to/image.jpg \
  --output outputs/segmentation --checkpoint checkpoints/seg_best.pt
```

Segmentation inference resizes to 512x512, applies the notebook preprocessing,
uses sigmoid + argmax, resizes to 4000x3000 by default (or `--output-size WxH`),
erodes once with a 3x3 kernel, and writes:

- `masks/` indexed PNGs
- `color/` colorized PNGs
- `overlays/` image overlays
- `predictions.json`

## Streamlit Demo

```bash
uv run streamlit run demo/app.py
```

The demo accepts an uploaded image, a checkpoint path, and a config path. It
shows classification probabilities or a segmentation overlay and lets you
download the indexed mask.

## Notebook Compatibility Notes

The implementation deliberately preserves notebook behavior where it affects
reproduction:

- Classification and segmentation pseudolabels are generated after the labeled
  training phase and used in the same epoch.
- Classification pseudolabels are indexed by dataset index.
- Segmentation pseudolabels are raw logits stored per batch and reused by batch
  index. The unlabeled DataLoader is not shuffled, matching the notebook.
- Segmentation semisupervised loss uses raw logits as targets in the equal-weight
  BCE + Dice loss.
- Segmentation validation uses the training augmentation pipeline because the
  notebook does.
- Classification submission output uses `pred ^ 1` by default.
- Classification alpha ramps from epoch 10 to 150 with `linspace(0, 1, 140)`.
  The paper reports an end epoch of 135; the notebook uses 150.
- Segmentation alpha ramps from epoch 15 to 655 with `linspace(0, 0.5, 640)`.
  In a 100-epoch run it reaches only about `0.0657`.

Documented deviations:

- Random seeds are fixed for Python, NumPy, PyTorch, and data splits.
- Cropped masks are resized with nearest-neighbor interpolation so class IDs are
  preserved. The notebook's cache used a default interpolation when resizing
  masks.
- The supervised segmentation stage starts from ImageNet encoder weights and
  uses Adam `lr=1e-3` as specified in the project plan. The original notebook
  resumes an external checkpoint and uses SGD `lr=0.01` for that stage.
- Pseudo-logit storage defaults to float32 to match the notebook. It can be
  changed with `legacy.segmentation_pseudo_store_dtype`.

## Tests And Lint

```bash
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src demo tests
```

Tests cover alpha schedules, BCE + Dice and raw-logit pseudo loss behavior,
classification and segmentation metrics, mask handling, pseudo-label indexing,
and a synthetic CPU smoke run for both training pipelines plus evaluation and
prediction.

## Project Layout

```text
configs/                  YAML configs for paper defaults and smoke tests
demo/app.py               Streamlit demo
src/floodnet/data/        download, validation, caching, datasets, transforms
src/floodnet/models/      ResNet18 and DeepLabV3+ builders
src/floodnet/engine/      training, evaluation, prediction, shared utilities
src/floodnet/losses.py    BCE + Dice loss
src/floodnet/metrics.py   F1, accuracy, mIoU, IoU, confusion matrices
tests/                    Unit and smoke tests
```
