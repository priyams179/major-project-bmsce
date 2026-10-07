#!/usr/bin/env bash
# Milestone 6: Round-3 Multi-Architecture Self-Training & SOTA Ensemble
# 1. Generate clean Round-3 pseudo-labels using ensemble teachers (m5, m4_dlv3p, m4_resnet) with TTA
# 2. Train 3 diverse students at scale 2 (256px) on real + pseudo data:
#    - Student 1: smp_deeplabv3plus_effb4
#    - Student 2: smp_unet_resnet50
#    - Student 3: smp_unet_mitb2
# 3. Comprehensive multi-architecture ensemble evaluation

set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p output models

echo "========================================================================"
echo "MILESTONE 6: ROUND-3 SELF-TRAINING & SOTA BENCHMARK"
echo "Started at: $(date)"
echo "========================================================================"

# Step 1 & 2: Student 1 (DeepLabV3+ EffB4) - Generates pseudo-labels first if not present
echo "=== [$(date +%H:%M:%S)] START Student 1: m6_dlv3p_s256_r3 ==="
$PY self_training.py \
    --teacher models/m5_selftrain_r2/best_model.pth \
              models/m4_dlv3p_s256/best_model.pth \
              models/m4_resnet_s256/best_model.pth \
    --pseudo-dir data/processed/l4s_pseudo_r3 \
    --norm split --scale 2 --loss bce_lovasz \
    --pos-thresh 0.70 --neg-thresh 0.30 \
    --epochs 22 --batch-size 8 --mixed-precision \
    --arch smp_deeplabv3plus_effb4 \
    --output-dir models/m6_dlv3p_s256_r3 \
    > output/log_m6_dlv3p_s256_r3.txt 2>&1
echo "=== [$(date +%H:%M:%S)] DONE  Student 1: m6_dlv3p_s256_r3 ==="

# Step 3: Student 2 (ResNet-50 UNet)
echo "=== [$(date +%H:%M:%S)] START Student 2: m6_resnet50_s256_r3 ==="
$PY self_training.py \
    --teacher models/m5_selftrain_r2/best_model.pth \
    --pseudo-dir data/processed/l4s_pseudo_r3 \
    --norm split --scale 2 --loss bce_lovasz \
    --epochs 22 --batch-size 8 --mixed-precision \
    --arch smp_unet_resnet50 \
    --output-dir models/m6_resnet50_s256_r3 \
    > output/log_m6_resnet50_s256_r3.txt 2>&1
echo "=== [$(date +%H:%M:%S)] DONE  Student 2: m6_resnet50_s256_r3 ==="

# Step 4: Student 3 (SegFormer MiT-B2)
echo "=== [$(date +%H:%M:%S)] START Student 3: m6_mitb2_s256_r3 ==="
$PY self_training.py \
    --teacher models/m5_selftrain_r2/best_model.pth \
    --pseudo-dir data/processed/l4s_pseudo_r3 \
    --norm split --scale 2 --loss bce_lovasz \
    --epochs 22 --batch-size 8 --mixed-precision \
    --arch smp_unet_mitb2 \
    --output-dir models/m6_mitb2_s256_r3 \
    > output/log_m6_mitb2_s256_r3.txt 2>&1
echo "=== [$(date +%H:%M:%S)] DONE  Student 3: m6_mitb2_s256_r3 ==="

# Step 5: Final Cross-Architecture Ensemble Evaluation
echo "=== [$(date +%H:%M:%S)] START Final SOTA Ensemble Evaluation ==="
$PY ensemble_eval.py --members \
    models/m6_dlv3p_s256_r3/best_model.pth \
    models/m6_resnet50_s256_r3/best_model.pth \
    models/m6_mitb2_s256_r3/best_model.pth \
    models/m5_selftrain_r2/best_model.pth \
    models/m4_dlv3p_s256/best_model.pth \
    models/m4_resnet_s256/best_model.pth \
    --weights 2.5 2.0 1.5 1.5 1.0 1.0 \
    --scale 2 \
    > output/log_m6_final_ensemble.txt 2>&1

cat output/log_m6_final_ensemble.txt

echo "========================================================================"
echo "MILESTONE 6 COMPLETED: $(date)"
echo "========================================================================"
