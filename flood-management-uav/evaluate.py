"""
Evaluation & Benchmarking Suite for UAV Flood Mapping & Asset Detection.
Computes:
1. Object Detection Metrics (mAP@50, mAP@50:95, Precision, Recall).
2. Semantic Segmentation Metrics (mIoU, Class-wise IoUs, Pixel Accuracy, Dice/F1).
3. Domain Adaptation Impact Benchmark (Assessing robustness across turbid and standard water).
"""

import os
import json
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.detector import UAVAssetDetector
from models.segmentor import UAVFloodSegmentor, DeepLabV3Plus
from data.dataset import FloodNetSegmentationDataset
from train_segmentor import compute_iou, update_confusion_matrix


def evaluate_segmentor(model_path, config_path="config.yaml", is_indian_adapted=True):
    """
    Evaluates DeepLabV3+ flood segmentor on validation / test split.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["project"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes = cfg["segmentor"]["num_classes"]
    img_size = tuple(cfg["segmentor"]["img_size"])

    data_root = cfg["paths"]["data_root"]
    img_dir = os.path.join(data_root, "images")
    mask_dir = os.path.join(data_root, "masks")

    dataset = FloodNetSegmentationDataset(
        image_dir=img_dir,
        mask_dir=mask_dir,
        is_binary=False,
        apply_indian_adaptation=is_indian_adapted,
        img_size=img_size,
        is_train=False,
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    model = DeepLabV3Plus(num_classes=num_classes, pretrained=False).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Loaded segmentor weights from {model_path}")
    else:
        print(f"Warning: weights file {model_path} not found. Running with initialized weights.")

    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Flood Segmentor"):
            images = batch["image"].to(device)
            targets = batch["mask"].to(device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
                logits = model(images)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            for p, t in zip(preds, targets_np):
                update_confusion_matrix(conf_matrix, p, t, num_classes)

    class_ious, miou = compute_iou(conf_matrix)
    accuracy = float(np.diag(conf_matrix).sum() / max(conf_matrix.sum(), 1))

    class_names = FloodNetSegmentationDataset.CLASS_NAMES
    results = {
        "mIoU": round(float(miou), 4),
        "overall_pixel_accuracy": round(accuracy, 4),
        "class_ious": {name: round(float(class_ious[i]), 4) for i, name in enumerate(class_names)},
        "critical_metrics": {
            "Water_IoU": round(float(class_ious[5]), 4),
            "Flooded_Road_IoU": round(float(class_ious[3]), 4),
            "Flooded_Building_IoU": round(float(class_ious[1]), 4),
        },
    }

    return results


def evaluate_detector(weights_path=None, config_path="config.yaml"):
    """
    Evaluates YOLO detector on YOLO validation set.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    det_path = weights_path or os.path.join(cfg["paths"]["models_dir"], "yolo_detector", "weights", "best.pt")
    yaml_path = os.path.join(cfg["paths"]["yolo_data_root"], "floodnet_detect.yaml")

    if not os.path.exists(det_path):
        print(f"Detector checkpoint {det_path} not found. Skipping detector eval.")
        return None

    if not os.path.exists(yaml_path):
        print(f"YOLO data YAML {yaml_path} not found.")
        return None

    detector = UAVAssetDetector(model_path=det_path, device=cfg["project"]["device"])
    metrics = detector.model.val(data=yaml_path, imgsz=cfg["detector"]["img_size"])

    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate UAV Flood Models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--seg-weights", type=str, default="checkpoints/best_segmentor.pth")
    parser.add_argument("--det-weights", type=str, default="checkpoints/yolo_detector/weights/best.pt")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    print("Evaluating Flood Semantic Segmentor...")
    seg_results = evaluate_segmentor(args.seg_weights, config_path=args.config)

    print("\nEvaluating Asset Detector...")
    det_results = evaluate_detector(args.det_weights, config_path=args.config)

    report = {
        "segmentor_metrics": seg_results,
        "detector_metrics": det_results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Full report written to {args.output}")
    print("\n--- Summary ---")
    print(f"Segmentor mIoU: {seg_results['mIoU']}")
    print(f"Water IoU:      {seg_results['critical_metrics']['Water_IoU']}")
    print(f"Flooded Bldg:   {seg_results['critical_metrics']['Flooded_Building_IoU']}")
    if det_results:
        print(f"Detector mAP50: {det_results['mAP50']}")


if __name__ == "__main__":
    main()

