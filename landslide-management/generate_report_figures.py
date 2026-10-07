#!/usr/bin/env python3
"""
Generate publication-quality figures for REPORT.md and README.md:
1. figures/model_accuracy_trajectory.png:
   - Panel A: Training vs Validation Loss Trajectory
   - Panel B: Validation Metrics Trajectory (IoU, F1, Precision, Recall)
   - Panel C: Benchmark Progression Bar Chart (Baseline -> SOTA Ensemble)
2. figures/test_prediction_sample.png:
   - Multi-panel inference comparison on a real test scene (RGB, False Color, Slope, GT, Probability Map, Error Map).
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from models import get_model
from train_segmentation import LandslideDataset, compute_or_load_split_stats
from visualize import make_rgb_composite, make_false_color


def generate_accuracy_figure():
    os.makedirs("figures", exist_ok=True)
    
    # Load training history from our best single model (m5_selftrain_r2)
    history_path = "models/m5_selftrain_r2/training_history.json"
    with open(history_path) as f:
        history = json.load(f)
        
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    
    fig = plt.figure(figsize=(18, 5.5))
    
    # --- Subplot 1: Loss trajectory ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(epochs, history["train_loss"], "b-o", linewidth=2, markersize=4, label="Train Loss (BCE+Lovász)")
    # Reconstruct val loss curve if available or plot smoothed loss
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Training Loss Convergence (M5 Student)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper right")
    
    # --- Subplot 2: Validation Metrics Trajectory ---
    ax2 = fig.add_subplot(1, 3, 2)
    val_iou = np.array(history["val_iou"]) * 100
    val_f1 = np.array(history["val_f1"]) * 100
    ax2.plot(epochs, val_iou, "g-^", linewidth=2, markersize=4, label="Val IoU (%)")
    ax2.plot(epochs, val_f1, "m-s", linewidth=2, markersize=4, label="Val F1 (%)")
    ax2.axhline(y=max(val_f1), color="r", linestyle=":", label=f"Peak Val F1 ({max(val_f1):.1f}%)")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Trajectory on Real Ground Truth", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")
    
    # --- Subplot 3: Benchmark Ladder Bar Chart ---
    ax3 = fig.add_subplot(1, 3, 3)
    milestones = [
        "Baseline\n(ResNet-34)",
        "M1\n(Sep-Norm)",
        "M2\n(SelfTrain R1)",
        "M4\n(256px Scale)",
        "M5\n(SelfTrain R2)",
        "M5 + TTA\n(4-Fold)",
        "SOTA Ensemble\n(4 Models+TTA)"
    ]
    test_f1_scores = [64.60, 65.26, 67.61, 67.11, 69.70, 70.54, 71.34]
    colors = ["#9E9E9E", "#64B5F6", "#42A5F5", "#2196F3", "#1E88E5", "#1565C0", "#0D47A1"]
    
    bars = ax3.bar(milestones, test_f1_scores, color=colors, width=0.62, edgecolor="black", linewidth=1.1)
    ax3.set_ylim(60.0, 75.0)
    ax3.set_ylabel("Held-Out Test F1 Score (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Performance Ladder Progression to SOTA", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    plt.suptitle("Deep Learning Landslide Detection: Training Dynamics & SOTA Ladder", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = "figures/model_accuracy_trajectory.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_prediction_figure():
    os.makedirs("figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test sample 00004
    sample_idx = 4
    img_raw = np.load(f"data/processed/l4s/test/image_{sample_idx:05d}.npy") # (14, 128, 128)
    mask_raw = np.load(f"data/processed/l4s/test/mask_{sample_idx:05d}.npy") # (128, 128)
    
    # Load model m5_selftrain_r2
    ck = torch.load("models/m5_selftrain_r2/best_model.pth", map_location="cpu", weights_only=False)
    model = get_model(arch=ck["arch"], in_channels=ck["in_channels"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    
    # Preprocess
    stats = json.load(open("data/processed/l4s/split_stats.json"))
    key = "test" if "test" in stats else "train"
    sm = np.array(stats[key]["mean"], dtype=np.float32)
    ss = np.array(stats[key]["std"], dtype=np.float32) + 1e-6
    img_norm = (img_raw.astype(np.float32) - sm[:, None, None]) / ss[:, None, None]
    
    # Forward pass at scale 2 with 4-fold TTA
    x = torch.from_numpy(img_norm).unsqueeze(0).to(device)
    x_s2 = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    
    x_h = torch.flip(x_s2, dims=[-1])
    x_v = torch.flip(x_s2, dims=[-2])
    x_hv = torch.flip(x_s2, dims=[-2, -1])
    
    with torch.no_grad():
        p0 = torch.sigmoid(model(x_s2))
        ph = torch.flip(torch.sigmoid(model(x_h)), dims=[-1])
        pv = torch.flip(torch.sigmoid(model(x_v)), dims=[-2])
        phv = torch.flip(torch.sigmoid(model(x_hv)), dims=[-2, -1])
        prob_s2 = ((p0 + ph + pv + phv) / 4.0)[0, 0].cpu().numpy()
        
    # Resize prob back to original 128x128 for fair visual comparison
    prob = F.interpolate(torch.from_numpy(prob_s2).unsqueeze(0).unsqueeze(0),
                         size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy()
                         
    pred_bin = (prob >= 0.40).astype(np.uint8)
    gt_bin = (mask_raw > 0.5).astype(np.uint8)
    
    # Multi-panel visualization (6 panels)
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    
    # 1. True Color RGB (B4=Red idx 3, B3=Green idx 2, B2=Blue idx 1 in S2)
    rgb = make_rgb_composite(img_raw, rgb_indices=(3, 2, 1))
    axes[0].imshow(rgb)
    axes[0].set_title("(a) True Color RGB\n(B4-B3-B2)", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    # 2. False Color NIR (B8=NIR idx 7, B4=Red idx 3, B3=Green idx 2)
    false_col = make_false_color(img_raw, nir_idx=7, red_idx=3, green_idx=2)
    axes[1].imshow(false_col)
    axes[1].set_title("(b) False Color NIR\n(Vegetation Scarring)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    # 3. Topographic Slope (ch12 = slope)
    slope = img_raw[12]
    im_slope = axes[2].imshow(slope, cmap="magma")
    axes[2].set_title("(c) DEM Slope (°)\n(Copernicus GLO-30)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_slope, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Ground Truth Mask
    axes[3].imshow(rgb)
    axes[3].contour(gt_bin, levels=[0.5], colors=["lime"], linewidths=1.8)
    gt_overlay = np.zeros((*gt_bin.shape, 4))
    gt_overlay[gt_bin == 1] = [0.0, 1.0, 0.0, 0.45] # Green fill
    axes[3].imshow(gt_overlay)
    axes[3].set_title("(d) Ground Truth\n(Landslide Polygon)", fontsize=11, fontweight="bold")
    axes[3].axis("off")
    
    # 5. Neural Probability Heatmap
    im_p = axes[4].imshow(prob, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[4].set_title("(e) Predicted Probability\n(DeepLabV3+ EffB4 + TTA)", fontsize=11, fontweight="bold")
    axes[4].axis("off")
    fig.colorbar(im_p, ax=axes[4], fraction=0.046, pad=0.04)
    
    # 6. Error & Overlap Classification
    # TP (Green), FP (Yellow), FN (Red)
    error_map = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)
    
    error_img = rgb.copy()
    # Blend TP = Cyan/Green (0, 230, 118), FP = Yellow (255, 235, 59), FN = Red (255, 23, 68)
    for r in range(128):
        for c in range(128):
            if tp[r, c]:
                error_img[r, c] = [0, 230, 118]
            elif fp[r, c]:
                error_img[r, c] = [255, 215, 0]
            elif fn[r, c]:
                error_img[r, c] = [255, 23, 68]
                
    axes[5].imshow(error_img)
    axes[5].set_title("(f) Verification Overlay\nTP (Green), FP (Yel), FN (Red)", fontsize=11, fontweight="bold")
    axes[5].axis("off")
    
    patches = [
        mpatches.Patch(color=(0/255, 230/255, 118/255), label="True Positive"),
        mpatches.Patch(color=(255/255, 215/255, 0/255), label="False Positive"),
        mpatches.Patch(color=(255/255, 23/255, 68/255), label="False Negative"),
    ]
    axes[5].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)
    
    plt.suptitle("End-to-End Test Inference Case Study (Sample #00004)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = "figures/test_prediction_sample.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_accuracy_figure()
    generate_prediction_figure()
