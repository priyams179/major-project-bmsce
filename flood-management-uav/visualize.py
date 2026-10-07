"""
Visualization Engine for UAV Flood Mapping & Asset Damage Assessment.
Produces publication-grade side-by-side diagnostic figures and composite disaster response maps.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color definitions (BGR for OpenCV, RGB for Matplotlib)
IMPACT_COLORS_BGR = {
    "Submerged / Inundated": (0, 0, 220),       # Red/Crimson
    "Perimeter Threatened": (0, 140, 255),      # Orange
    "Isolated / Cut-off Island": (0, 215, 255), # Bright Gold / Amber
    "Safe / Dry": (0, 200, 0),                  # Sea Green
    "Submerged / Trapped": (255, 200, 0),       # Cyan / Electric Blue
    "Operational / Dry": (0, 215, 255),         # Gold
}

SEGMENTATION_PALETTE = {
    0: (40, 40, 40),       # Background
    1: (220, 20, 60),      # Building Flooded (Crimson)
    2: (46, 139, 87),      # Building Non-Flooded (Sea Green)
    3: (255, 140, 0),      # Road Flooded (Dark Orange)
    4: (160, 160, 160),    # Road Non-Flooded (Gray)
    5: (30, 144, 255),     # Water (Dodger Blue)
    6: (34, 139, 34),      # Tree (Forest Green)
    7: (255, 215, 0),      # Vehicle (Gold)
    8: (0, 255, 255),      # Pool (Cyan)
    9: (154, 205, 50),     # Grass (Yellow Green)
}


def colorize_segmentation_mask(mask):
    """Converts a 2D integer class mask into an RGB color image."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in SEGMENTATION_PALETTE.items():
        color_img[mask == class_id] = color
    return color_img


def generate_composite_flood_map(
    rgb_image,
    flood_mask,
    pred_mask,
    assets,
    report=None,
    save_path="outputs/composite_flood_map.png",
):
    """
    Renders a composite disaster overlay:
    1. Base UAV RGB image
    2. Translucent flood inundation layer (cyan/blue)
    3. Asset bounding boxes color-coded by flood risk
    4. Executive HUD scoreboard with statistics
    """
    h, w, _ = rgb_image.shape
    canvas_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # 1. Overlay translucent flood mask (Dodger Blue: BGR 255, 144, 30)
    flood_overlay = canvas_bgr.copy()
    flood_color = np.array([255, 144, 30], dtype=np.uint8)
    flood_overlay[flood_mask == 1] = cv2.addWeighted(
        flood_overlay[flood_mask == 1], 0.55,
        np.full_like(flood_overlay[flood_mask == 1], flood_color), 0.45, 0
    )
    canvas_bgr = flood_overlay

    # 2. Draw Asset Bounding Boxes
    for asset in assets:
        x1, y1, x2, y2 = [int(v) for v in asset["box"]]
        status = asset["impact_status"]
        color = IMPACT_COLORS_BGR.get(status, (255, 255, 255))

        # Thick border
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 2)

        # Label tag
        sub_pct = asset["submergence_ratio"] * 100
        tag = f"{asset['class_name']}: {sub_pct:.0f}% submerged"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        cv2.rectangle(canvas_bgr, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas_bgr,
            tag,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255) if color != (255, 215, 0) else (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # 3. Draw Disaster HUD / Scoreboard (Top Right Corner)
    if report is not None:
        hud_w, hud_h = 360, 200
        hud_x1 = w - hud_w - 20
        hud_y1 = 20
        hud_x2 = hud_x1 + hud_w
        hud_y2 = hud_y1 + hud_h

        # Semi-transparent dark slate backdrop
        hud_sub = canvas_bgr[hud_y1:hud_y2, hud_x1:hud_x2]
        dark_bg = np.zeros_like(hud_sub)
        canvas_bgr[hud_y1:hud_y2, hud_x1:hud_x2] = cv2.addWeighted(hud_sub, 0.25, dark_bg, 0.75, 0)
        cv2.rectangle(canvas_bgr, (hud_x1, hud_y1), (hud_x2, hud_y2), (200, 200, 200), 2)

        # Title
        cv2.putText(
            canvas_bgr,
            "UAV FLOOD DISASTER ASSESSMENT",
            (hud_x1 + 15, hud_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Severity line
        sev_color = (0, 0, 255) if "Level 3" in report["overall_severity"] else (0, 165, 255)
        cv2.putText(
            canvas_bgr,
            f"Severity: {report['overall_severity']}",
            (hud_x1 + 15, hud_y1 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            sev_color,
            1,
            cv2.LINE_AA,
        )

        # Stats lines
        b_sub = report["buildings"]["submerged"]
        b_tot = report["buildings"]["total"]
        v_sub = report["vehicles"]["submerged"]
        v_tot = report["vehicles"]["total"]
        f_pct = report["flood_inundation_percentage"]

        lines = [
            f"Flood Inundation Area: {f_pct:.1f}%",
            f"Buildings Submerged:   {b_sub} / {b_tot}",
            f"Buildings Isolated:    {report['buildings'].get('isolated', 0)} / {b_tot}",
            f"Buildings Threatened:  {report['buildings']['threatened']} / {b_tot}",
            f"Vehicles Trapped:      {v_sub} / {v_tot}",
            f"Total Assets Tracked:  {report['total_detected_assets']}",
        ]

        for idx, line in enumerate(lines):
            cv2.putText(
                canvas_bgr,
                line,
                (hud_x1 + 15, hud_y1 + 80 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, canvas_bgr)
    return canvas_bgr


def generate_side_by_side_comparison(
    rgb_image,
    pred_mask,
    flood_mask,
    assets,
    save_path="outputs/side_by_side_analysis.png",
):
    """
    Renders a 3-panel side-by-side diagnostic figure using Matplotlib:
    Panel 1: Original UAV Aerial RGB
    Panel 2: 10-Class Semantic Flood & Damage Map
    Panel 3: Physical Asset Detection with Flood Risk Annotations
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=150)

    # Panel 1: Original UAV Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("1. Raw UAV Aerial RGB Input", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Semantic Segmentation Map
    color_mask = colorize_segmentation_mask(pred_mask)
    axes[1].imshow(color_mask)
    axes[1].set_title("2. DeepLabV3+ Flood Semantic Map", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Legend for Panel 2
    legend_patches = [
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[5]) / 255.0, label="Water"),
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[1]) / 255.0, label="Building Flooded"),
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[2]) / 255.0, label="Building Dry"),
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[3]) / 255.0, label="Road Flooded"),
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[7]) / 255.0, label="Vehicle"),
        mpatches.Patch(color=np.array(SEGMENTATION_PALETTE[9]) / 255.0, label="Vegetation/Grass"),
    ]
    axes[1].legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.85)

    # Panel 3: Asset Detection Overlay
    det_canvas = rgb_image.copy()
    for asset in assets:
        x1, y1, x2, y2 = [int(v) for v in asset["box"]]
        status = asset["impact_status"]
        color_bgr = IMPACT_COLORS_BGR.get(status, (255, 255, 255))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        cv2.rectangle(det_canvas, (x1, y1), (x2, y2), color_rgb, 3)

    axes[2].imshow(det_canvas)
    axes[2].set_title("3. Asset Detection & Risk Classification", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    # Legend for Panel 3
    risk_patches = [
        mpatches.Patch(color=np.array([220, 0, 0]) / 255.0, label="Submerged Building (>15%)"),
        mpatches.Patch(color=np.array([255, 140, 0]) / 255.0, label="Threatened Building (5-15%)"),
        mpatches.Patch(color=np.array([255, 215, 0]) / 255.0, label="Isolated / Cut-off Island"),
        mpatches.Patch(color=np.array([0, 200, 0]) / 255.0, label="Safe / Dry Building"),
        mpatches.Patch(color=np.array([0, 200, 255]) / 255.0, label="Submerged Vehicle"),
    ]
    axes[2].legend(handles=risk_patches, loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path

