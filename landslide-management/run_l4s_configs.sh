#!/usr/bin/env bash
# Run the 4 approved L4S configs sequentially (plan: tight loop, report at end).
# Selection on val IoU; the ranking decision uses val only.
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

run l4s_bce_pw      --arch resnet34_unet --loss bce_dice --pos-weight 42 --output-dir models/l4s_bce_pw
run l4s_unet_b64    --arch unet --base-features 64 --loss bce_dice --output-dir models/l4s_unet_b64_bce
run l4s_bce_dropout --arch resnet34_unet --loss bce_dice --decoder-dropout 0.2 --output-dir models/l4s_bce_dropout
run l4s_combined    --arch resnet34_unet --loss combined --output-dir models/l4s_combined

echo "=== ALL CONFIGS FINISHED $(date +%H:%M:%S) ==="
