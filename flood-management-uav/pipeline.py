"""
End-to-End UAV Flood Mapping & Asset Impact Assessment Pipeline.

Workflow:
1. Ingests high-resolution UAV aerial imagery.
2. Stage 1: Aerial Object Detection (YOLOv8/v11) localizes and classifies
   all Buildings and Vehicles/Cars with precise bounding boxes.
3. Stage 2: Flood Semantic Segmentation (DeepLabV3+) generates pixel-level
   flood inundation maps (Water, Flooded Roads, Flooded Structures).
4. Stage 3: Spatial Overlay & Impact Analysis:
   - Evaluates submergence ratios for each individual structure.
   - Flags submerged vs dry vehicles.
   - Calculates total flooded area, affected building tally, and road accessibility.
5. Produces comprehensive disaster response reports and composite visual overlays.
"""

import os
import json
import cv2
import numpy as np
import yaml
import torch

from models.detector import UAVAssetDetector
from models.segmentor import UAVFloodSegmentor


class UAVFloodMappingPipeline:
    """
    Unified multi-stage pipeline for UAV asset detection and flood mapping.
    """

    def __init__(self, config_path="config.yaml", detector_weights=None, segmentor_weights=None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = self.cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
        print(f"Initializing UAV Flood Mapping Pipeline on {self.device}...")

        # Initialize Stage 1: Asset Detector
        det_path = detector_weights or os.path.join(
            self.cfg["paths"]["models_dir"], "yolo_detector", "weights", "best.pt"
        )
        if not os.path.exists(det_path):
            print(f"Custom detector weights not found at {det_path}. Using base {self.cfg['detector']['model_type']}...")
            det_path = self.cfg["detector"]["model_type"]

        self.detector = UAVAssetDetector(model_path=det_path, device=self.device)

        # Initialize Stage 2: Flood Segmentor
        seg_path = segmentor_weights or os.path.join(
            self.cfg["paths"]["models_dir"], "best_segmentor.pth"
        )
        if not os.path.exists(seg_path):
            print(f"Segmentation weights not found at {seg_path}. Model will initialize with backbone weights.")
            seg_path = None

        self.segmentor = UAVFloodSegmentor(
            weights_path=seg_path,
            num_classes=self.cfg["segmentor"]["num_classes"],
            device=self.device,
        )

        # Thresholds
        impact_cfg = self.cfg.get("impact_assessment", {})
        self.bldg_submerged_thresh = impact_cfg.get("submergence_threshold_building", 0.15)
        self.bldg_threat_thresh = impact_cfg.get("threatened_threshold_building", 0.05)
        self.veh_submerged_thresh = impact_cfg.get("vehicle_submerged_threshold", 0.10)

    def process_image(self, image_input, output_dir=None, save_vis=True):
        """
        Executes the two-stage flood mapping and asset assessment on a UAV image.

        Args:
            image_input: path to image or RGB numpy array.
            output_dir: directory to save output results.
            save_vis: whether to generate and save visual overlay images.

        Returns:
            dict containing:
                - detections: list of asset bounding boxes with impact stats
                - segmentation_mask: 2D array of class IDs
                - flood_mask: 2D binary inundation array (1=flooded, 0=dry)
                - disaster_report: structured impact assessment statistics
        """
        if isinstance(image_input, str):
            orig_bgr = cv2.imread(image_input)
            if orig_bgr is None:
                raise FileNotFoundError(f"Could not open image file: {image_input}")
            rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            img_name = os.path.splitext(os.path.basename(image_input))[0]
        else:
            rgb = image_input.copy()
            orig_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img_name = "uav_sample"

        h, w, _ = rgb.shape

        # -------------------------------------------------------------
        # STAGE 1: Asset Detection & Classification (Buildings & Cars)
        # -------------------------------------------------------------
        print(f"Stage 1: Running UAV Asset Detection on image ({w}x{h})...")
        use_tiling = self.cfg["detector"]["tiling"]["enabled"] and (w > 1200 or h > 1200)
        if use_tiling:
            tile_sz = self.cfg["detector"]["tiling"]["tile_size"]
            overlap = self.cfg["detector"]["tiling"]["overlap"]
            raw_detections = self.detector.predict_tiled(
                rgb,
                tile_size=tile_sz,
                overlap=overlap,
                conf=self.cfg["detector"]["conf_threshold"],
            )
        else:
            raw_detections = self.detector.predict(
                rgb,
                conf=self.cfg["detector"]["conf_threshold"],
            )

        print(f"Stage 1 Complete: Detected {len(raw_detections)} physical assets.")

        # -------------------------------------------------------------
        # STAGE 2: Smoothed Flood Likelihood & Inundation Mapping
        # -------------------------------------------------------------
        print("Stage 2: Running UAV Flood Semantic Mapping with Spatial Regularization...")
        smooth_cfg = self.cfg.get("impact_assessment", {}).get("smoothing", {})
        pred_mask, flood_mask, flood_prob = self.segmentor.predict_smoothed_flood(
            rgb,
            img_size=tuple(self.cfg["segmentor"]["img_size"]),
            gaussian_ksize=smooth_cfg.get("gaussian_ksize", 15),
            morph_ksize=smooth_cfg.get("morph_ksize", 7),
            prob_threshold=smooth_cfg.get("water_prob_thresh", 0.40),
        )
        print("Stage 2 Complete: Continuous regularized flood extent mapped.")

        # -------------------------------------------------------------
        # STAGE 3: Object-Centric Spatial Impact Reasoning & Road Passability
        # -------------------------------------------------------------
        print("Stage 3: Computing Object-Centric Spatial Overlap & Access Corridor Severance...")
        analyzed_assets = []
        building_stats = {"total": 0, "submerged": 0, "threatened": 0, "isolated": 0, "dry": 0}
        vehicle_stats = {"total": 0, "submerged": 0, "dry": 0}

        # Road Network Analysis from Semantic Output
        road_flooded_pixels = int(np.sum(pred_mask == 3))
        road_dry_pixels = int(np.sum(pred_mask == 4))
        total_road_pixels = road_flooded_pixels + road_dry_pixels
        road_submerged_pct = (road_flooded_pixels / max(1, total_road_pixels)) * 100.0 if total_road_pixels > 0 else 0.0
        road_severance_thresh = self.cfg.get("impact_assessment", {}).get("road_severance_threshold", 0.20) * 100.0
        is_corridor_severed = road_submerged_pct >= road_severance_thresh

        buffer_px = self.cfg.get("impact_assessment", {}).get("building_threat_buffer_px", 25)

        for det in raw_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            box_area = box_w * box_h

            # 1. Footprint Submergence
            box_flood_slice = flood_mask[y1:y2, x1:x2]
            flooded_pixels = int(np.sum(box_flood_slice))
            submergence_ratio = float(flooded_pixels / box_area)

            # 2. Perimeter Buffer Water Encroachment
            px1, py1 = max(0, x1 - buffer_px), max(0, y1 - buffer_px)
            px2, py2 = min(w, x2 + buffer_px), min(h, y2 + buffer_px)
            peri_flood_slice = flood_mask[py1:py2, px1:px2]
            peri_total = max(1, (px2 - px1) * (py2 - py1) - box_area)
            peri_water = int(np.sum(peri_flood_slice)) - flooded_pixels
            perimeter_threat_ratio = float(max(0, peri_water) / peri_total)

            cid = det["class_id"]
            cname = det["class_name"].lower()
            is_building = ("building" in cname) or (cid in [0, 1])
            is_vehicle = ("vehicle" in cname) or ("car" in cname) or (cid == 2)

            impact_status = "Safe / Dry"
            priority_rank = 4

            if is_building:
                building_stats["total"] += 1
                if submergence_ratio >= self.bldg_submerged_thresh:
                    impact_status = "Submerged / Inundated"
                    priority_rank = 1
                    building_stats["submerged"] += 1
                elif perimeter_threat_ratio >= 0.35 and is_corridor_severed:
                    impact_status = "Isolated / Cut-off Island"
                    priority_rank = 2
                    building_stats["isolated"] += 1
                elif submergence_ratio >= self.bldg_threat_thresh or perimeter_threat_ratio >= 0.20:
                    impact_status = "Perimeter Threatened"
                    priority_rank = 3
                    building_stats["threatened"] += 1
                else:
                    impact_status = "Safe / Dry"
                    priority_rank = 5
                    building_stats["dry"] += 1
            elif is_vehicle:
                vehicle_stats["total"] += 1
                if submergence_ratio >= self.veh_submerged_thresh:
                    impact_status = "Submerged / Trapped"
                    priority_rank = 2
                    vehicle_stats["submerged"] += 1
                else:
                    impact_status = "Operational / Dry"
                    priority_rank = 5
                    vehicle_stats["dry"] += 1

            asset_record = {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "box": [x1, y1, x2, y2],
                "confidence": det["confidence"],
                "submergence_ratio": round(submergence_ratio, 4),
                "perimeter_threat_ratio": round(perimeter_threat_ratio, 4),
                "impact_status": impact_status,
                "priority_rank": priority_rank,
            }
            analyzed_assets.append(asset_record)

        # Sort analyzed assets by triage priority rank
        analyzed_assets.sort(key=lambda a: (a["priority_rank"], -a["submergence_ratio"]))

        # Scene-Level Disaster Metrics
        total_pixels = h * w
        flood_pixels = int(np.sum(flood_mask))
        flood_percentage = (flood_pixels / total_pixels) * 100.0

        # Class area breakdown
        class_areas = {}
        for cid, name in self.cfg["classes"]["segmentation"].items():
            cnt = int(np.sum(pred_mask == cid))
            class_areas[name] = round((cnt / total_pixels) * 100.0, 2)

        # Indian Disaster Severity Classification (NDRF / SDMA reference)
        sub_ratio = (building_stats["submerged"] + building_stats["isolated"]) / max(1, building_stats["total"])
        if flood_percentage > 25.0 or (building_stats["total"] > 0 and sub_ratio > 0.35):
            severity = "Level 3 - Critical / Red Alert"
        elif flood_percentage > 10.0 or (building_stats["total"] > 0 and sub_ratio > 0.15) or is_corridor_severed:
            severity = "Level 2 - Moderate / Orange Alert"
        else:
            severity = "Level 1 - Low / Yellow Alert"

        disaster_report = {
            "scene_name": img_name,
            "dimensions": {"width": w, "height": h},
            "overall_severity": severity,
            "flood_inundation_percentage": round(flood_percentage, 2),
            "land_cover_breakdown_percent": class_areas,
            "road_corridor_status": {
                "total_road_coverage_percent": round((total_road_pixels / total_pixels) * 100.0, 2),
                "road_submergence_percent": round(road_submerged_pct, 2),
                "corridor_severed": is_corridor_severed,
                "navigation_status": "Impassable / Corridors Cut Off" if is_corridor_severed else "Passable Evacuation Corridor",
            },
            "buildings": building_stats,
            "vehicles": vehicle_stats,
            "total_detected_assets": len(analyzed_assets),
            "triage_queue_summary": {
                "priority_1_immediate_rescue": sum(1 for a in analyzed_assets if a["priority_rank"] == 1),
                "priority_2_urgent_aid_or_trapped": sum(1 for a in analyzed_assets if a["priority_rank"] == 2),
                "priority_3_perimeter_threatened": sum(1 for a in analyzed_assets if a["priority_rank"] == 3),
                "priority_safe": sum(1 for a in analyzed_assets if a["priority_rank"] >= 4),
            },
        }

        # -------------------------------------------------------------
        # STAGE 4: Visualization and Export
        # -------------------------------------------------------------
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            # 1. Save Report JSON
            report_path = os.path.join(output_dir, f"{img_name}_disaster_report.json")
            with open(report_path, "w") as f:
                json.dump(disaster_report, f, indent=2)

            # 2. Save Binary Flood Mask
            flood_mask_path = os.path.join(output_dir, f"{img_name}_flood_mask.png")
            cv2.imwrite(flood_mask_path, flood_mask * 255)

            # 3. Save Composite Overlay
            if save_vis:
                from visualize import generate_composite_flood_map
                vis_path = os.path.join(output_dir, f"{img_name}_flood_mapping_result.png")
                generate_composite_flood_map(
                    rgb_image=rgb,
                    flood_mask=flood_mask,
                    pred_mask=pred_mask,
                    assets=analyzed_assets,
                    report=disaster_report,
                    save_path=vis_path,
                )
                print(f"Saved comprehensive visualization to: {vis_path}")

        return {
            "assets": analyzed_assets,
            "pred_mask": pred_mask,
            "flood_mask": flood_mask,
            "flood_prob": flood_prob,
            "report": disaster_report,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run UAV Flood Mapping Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Input UAV image path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    pipeline = UAVFloodMappingPipeline(config_path=args.config)
    results = pipeline.process_image(args.image, output_dir=args.output_dir)
    print("\n=== UAV Flood Mapping Disaster Summary ===")
    print(json.dumps(results["report"], indent=2))

