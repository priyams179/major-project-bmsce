"""
UAV Aerial Asset Detector Wrapper (YOLOv8 / YOLOv11).
Specialized in detecting and bounding-box localizing:
  0: Building-Flooded
  1: Building-Non-Flooded
  2: Vehicle

Supports standard inference and Tiled / Sliced Inference for high-resolution UAV imagery.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class UAVAssetDetector:
    """
    Detector for UAV aerial assets (Buildings, Vehicles/Cars).
    """

    CLASS_NAMES = {
        0: "Building",
        1: "Vehicle",
    }

    CLASS_COLORS = {
        0: (0, 180, 0),    # Green / Sea Green for Building (BGR)
        1: (0, 215, 255),  # Gold / Amber for vehicle
    }

    def __init__(self, model_path="yolov8s.pt", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_path = model_path
        print(f"Loading YOLO detector from {model_path} onto {self.device}...")
        self.model = YOLO(model_path)
        if hasattr(self.model, "names") and self.model.names:
            self.CLASS_NAMES = {int(k): v for k, v in self.model.names.items()}

    def train(
        self,
        data_yaml="data/yolo_detection_large/floodnet_detect.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        project="checkpoints",
        name="yolo_detector_large",
        lr0=0.002,
    ):
        """
        Fine-tunes the YOLO model on the UAV dataset.
        """
        print(f"Starting YOLO detector training on {data_yaml} for {epochs} epochs...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=os.path.abspath(project),
            name=name,
            lr0=lr0,
            plots=True,
            save=True,
            val=True,
            exist_ok=True,
        )
        return results

    def predict(self, image, conf=0.25, iou=0.45):
        """
        Performs standard detection inference on an input image.

        Args:
            image: numpy array (RGB or BGR) or path string.
            conf: Confidence score threshold.
            iou: NMS IoU threshold.

        Returns:
            List of dicts: [
                {
                    "class_id": int,
                    "class_name": str,
                    "box": [x1, y1, x2, y2], (pixel coordinates)
                    "confidence": float
                }, ...
            ]
        """
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                cid = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())

                # Map to target class name if recognized
                cname = self.CLASS_NAMES.get(cid, self.model.names.get(cid, f"class_{cid}"))

                detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "box": [round(c, 2) for c in xyxy],
                    "confidence": round(score, 4),
                })

        return detections

    def predict_tiled(self, image, tile_size=1024, overlap=0.2, conf=0.25, iou=0.45):
        """
        Tiled / Sliced Inference for high-resolution UAV images (e.g. 4000x3000).
        Slides a window across the image, predicts locally, and merges with global NMS.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape
        stride = int(tile_size * (1.0 - overlap))

        raw_boxes = []
        raw_scores = []
        raw_classes = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)

                tile = image[y_start:y_end, x_start:x_end]
                tile_dets = self.predict(tile, conf=conf, iou=iou)

                for det in tile_dets:
                    bx1, by1, bx2, by2 = det["box"]
                    # Offset to global image coordinates
                    gx1 = bx1 + x_start
                    gy1 = by1 + y_start
                    gx2 = bx2 + x_start
                    gy2 = by2 + y_start

                    raw_boxes.append([gx1, gy1, gx2, gy2])
                    raw_scores.append(det["confidence"])
                    raw_classes.append(det["class_id"])

        if len(raw_boxes) == 0:
            return []

        # Global Non-Maximum Suppression (NMS)
        boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(raw_classes, dtype=torch.long)

        # Batch NMS per class
        keep_indices = []
        for unique_cls in torch.unique(classes_tensor):
            mask = classes_tensor == unique_cls
            cls_boxes = boxes_tensor[mask]
            cls_scores = scores_tensor[mask]
            cls_orig_idx = torch.nonzero(mask).squeeze(1)

            from torchvision.ops import nms
            kept = nms(cls_boxes, cls_scores, iou)
            keep_indices.extend(cls_orig_idx[kept].tolist())

        merged_detections = []
        for idx in keep_indices:
            cid = raw_classes[idx]
            cname = self.CLASS_NAMES.get(cid, self.model.names.get(cid, f"class_{cid}"))
            merged_detections.append({
                "class_id": cid,
                "class_name": cname,
                "box": [round(c, 2) for c in raw_boxes[idx]],
                "confidence": round(raw_scores[idx], 4),
            })

        return merged_detections

    def draw_boxes(self, image, detections, draw_confidence=True):
        """
        Draws colored bounding boxes with labels onto the image.
        """
        canvas = image.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cid = det["class_id"]
            color = self.CLASS_COLORS.get(cid, (255, 255, 0))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = det["class_name"]
            if draw_confidence:
                label += f" {det['confidence']:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

