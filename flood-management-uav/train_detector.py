"""
Training Script for UAV Aerial Asset Detector (Buildings and Vehicles/Cars).
Uses YOLOv8 / YOLOv11 for high-speed, precise object localization on drone imagery.
"""

import os
import argparse
import yaml
from models.detector import UAVAssetDetector
from data.prepare_detection_data import prepare_yolo_dataset


def main():
    parser = argparse.ArgumentParser(description="Train UAV Asset Detector (YOLO)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Base YOLO weights (e.g. yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg["detector"]
    model_name = args.model or det_cfg["model_type"]
    epochs = args.epochs or det_cfg["epochs"]
    batch_size = args.batch_size or det_cfg["batch_size"]
    imgsz = args.imgsz or det_cfg["img_size"]
    lr0 = det_cfg.get("lr0", 0.01)

    yolo_data_dir = cfg["paths"]["yolo_data_root"]
    yaml_path = os.path.join(yolo_data_dir, "floodnet_detect.yaml")

    # If YOLO dataset does not exist, prepare it from raw FloodNet data
    if not os.path.exists(yaml_path):
        print(f"YOLO dataset config not found at {yaml_path}. Preparing dataset...")
        raw_data_root = cfg["paths"]["data_root"]
        img_dir = os.path.join(raw_data_root, "images")
        mask_dir = os.path.join(raw_data_root, "masks")

        if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
            print("FloodNet raw data not found. Generating sample data...")
            from data.download_floodnet import create_verification_samples
            create_verification_samples(base_dir=raw_data_root, num_samples=24)

        prepare_yolo_dataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            output_yolo_dir=yolo_data_dir,
            val_split=0.2,
            min_area=30,
        )

    # Initialize and train detector
    detector = UAVAssetDetector(model_path=model_name, device=cfg["project"]["device"])

    results = detector.train(
        data_yaml=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=cfg["paths"]["models_dir"],
        name="yolo_detector",
        lr0=lr0,
    )

    print("\nDetector training completed.")
    best_pt = os.path.join(cfg["paths"]["models_dir"], "yolo_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"Best detector weights saved to: {best_pt}")


if __name__ == "__main__":
    main()

