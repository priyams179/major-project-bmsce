#!/usr/bin/env bash
# Milestone 1 of the SOTA ladder: normalization + loss levers on the
# baseline architecture. Selection on val only.
cd "$(dirname "$0")"

COMMON="--data-dir data/processed/l4s --in-channels 14 --epochs 30"

run() {
  local name="$1"; shift
  echo "=== [$(date +%H:%M:%S)] START $name : $* ==="
  if .venv/bin/python train_segmentation.py "$@" $COMMON \
      > "output/log_${name}.txt" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] DONE  $name ==="
  else
    echo "=== [$(date +%H:%M:%S)] FAILED $name (see output/log_${name}.txt) ==="
  fi
}

# M1a: isolate separate normalization (biggest single gain in 1st-place solution)
run m1_sepnorm      --arch resnet34_unet --loss bce_dice   --norm split --output-dir models/m1_sepnorm
# M1b: sep-norm + BCE+Lovasz (2nd-place team's combination)
run m1_sepnorm_lov  --arch resnet34_unet --loss bce_lovasz --norm split --output-dir models/m1_sepnorm_lov

echo "=== MILESTONE 1 FINISHED $(date +%H:%M:%S) ==="
