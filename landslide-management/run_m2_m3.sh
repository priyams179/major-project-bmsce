#!/usr/bin/env bash
# Milestones 2+3 of the SOTA ladder:
#   M2: self-training (teacher = M1 winner) -> students A/B
#   M3: pretrained smp encoders with the M1 loss/norm recipe
# Selection on val only; each script reports its own single test eval.
cd "$(dirname "$0")"

COMMON="--data-dir data/processed/l4s --in-channels 14"

run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name : $* ==="
  if .venv/bin/python "$@" > "output/log_${name}.txt" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $name ==="
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (see output/log_${name}.txt) ==="
  fi
}

# ---- Milestone 2: self-training (teacher = M1 winner) ----
run m2_selftrain  self_training.py \
  --teacher models/m1_sepnorm_lov/best_model.pth \
  --norm split --loss bce_lovasz --epochs 30 --mixed-precision \
  --output-dir models/m2_selftrain

# ---- Milestone 3: stronger pretrained encoders (M1 recipe) ----
run m3_effb4      train_segmentation.py --arch smp_unet_effb4 \
  --loss bce_lovasz --norm split --epochs 30 --mixed-precision \
  --data-dir data/processed/l4s --in-channels 14 \
  --output-dir models/m3_effb4

run m3_mitb2      train_segmentation.py --arch smp_unet_mitb2 \
  --loss bce_lovasz --norm split --epochs 30 --mixed-precision \
  --data-dir data/processed/l4s --in-channels 14 \
  --output-dir models/m3_mitb2

run m3_dlv3p      train_segmentation.py --arch smp_deeplabv3plus_effb4 \
  --loss bce_lovasz --norm split --epochs 30 --mixed-precision \
  --data-dir data/processed/l4s --in-channels 14 \
  --output-dir models/m3_dlv3p

echo "=== MILESTONES 2+3 FINISHED $(date +%H:%M:%S) ==="
