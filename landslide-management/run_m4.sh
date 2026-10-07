#!/usr/bin/env bash
# Milestone 4 of the SOTA ladder:
#   M4a: scale promotion 128->256 px (1st-place lever) on two diverse backbones
#   M4b: cross-architecture ensemble evaluation of the best checkpoints
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== [$(date +%H:%M:%S)] START m4_resnet_s256 ==="
if $PY train_segmentation.py --arch resnet34_unet --loss bce_lovasz \
    --norm split --scale 2 --data-dir data/processed/l4s --in-channels 14 \
    --epochs 20 --batch-size 8 --mixed-precision \
    --output-dir models/m4_resnet_s256 > output/log_m4_resnet_s256.txt 2>&1; then
  echo "=== [$(date +%H:%M:%S)] DONE  m4_resnet_s256 ==="
else
  echo "=== [$(date +%H:%M:%S)] FAILED m4_resnet_s256 ==="
fi

echo "=== [$(date +%H:%M:%S)] START m4_dlv3p_s256 ==="
if $PY train_segmentation.py --arch smp_deeplabv3plus_effb4 --loss bce_lovasz \
    --norm split --scale 2 --data-dir data/processed/l4s --in-channels 14 \
    --epochs 20 --batch-size 8 --mixed-precision \
    --output-dir models/m4_dlv3p_s256 > output/log_m4_dlv3p_s256.txt 2>&1; then
  echo "=== [$(date +%H:%M:%S)] DONE  m4_dlv3p_s256 ==="
else
  echo "=== [$(date +%H:%M:%S)] FAILED m4_dlv3p_s256 ==="
fi

echo "=== [$(date +%H:%M:%S)] START ensemble eval (elected on val) ==="
$PY ensemble_eval.py --data-dir data/processed/l4s --members \
  models/m3_dlv3p/best_model.pth \
  models/m2_selftrain/best_model.pth \
  models/m3_mitb2/best_model.pth \
  > output/log_ensemble_eval.txt 2>&1
cat output/log_ensemble_eval.txt

echo "=== MILESTONE 4 FINISHED $(date +%H:%M:%S) ==="
