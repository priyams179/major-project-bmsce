#!/usr/bin/env bash
# Milestone 5: progressive refinement round 2 (1st-place recipe).
# Teacher = current best-val model (m4_dlv3p_s256, val IoU 58.62%).
# Fresh pseudo-labels for the new teacher's probability maps, student retrained
# with the same recipe, validated on REAL val labels, one test eval at the end.
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== [$(date +%H:%M:%S)] START m5_round2_selftrain ==="
if $PY self_training.py \
    --teacher models/m4_dlv3p_s256/best_model.pth \
    --pseudo-dir data/processed/l4s_pseudo_r2 \
    --norm split --scale 2 --loss bce_lovasz \
    --epochs 20 --batch-size 8 --mixed-precision \
    --output-dir models/m5_selftrain_r2 > output/log_m5_selftrain_r2.txt 2>&1; then
  echo "=== [$(date +%H:%M:%S)] DONE  m5_round2_selftrain ==="
else
  echo "=== [$(date +%H:%M:%S)] FAILED m5_round2_selftrain ==="
fi
