"""Classification and segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from floodnet.constants import CLASS_NAMES, NUM_CLASSES


class ClassificationMetrics:
    def __init__(self) -> None:
        self.true: list[int] = []
        self.predicted: list[int] = []

    def update(self, logits: Any, labels: Any) -> None:
        predictions = logits.argmax(dim=1).detach().cpu().tolist()
        self.predicted.extend(int(value) for value in predictions)
        self.true.extend(int(value) for value in labels.detach().cpu().tolist())

    def compute(self) -> dict[str, Any]:
        if not self.true:
            return {"accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion_matrix": []}
        labels = [0, 1]
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true,
            self.predicted,
            labels=labels,
            zero_division=0,
        )
        per_class = {
            "Non-Flooded" if index == 0 else "Flooded": {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        }
        return {
            "accuracy": float(accuracy_score(self.true, self.predicted)),
            "macro_f1": float(f1_score(self.true, self.predicted, average="macro", zero_division=0)),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(self.true, self.predicted, labels=labels).tolist(),
            "samples": len(self.true),
        }


class SegmentationMetrics:
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: Any, targets: Any) -> None:
        predicted = predictions.detach().cpu().numpy().reshape(-1)
        target = targets.detach().cpu().numpy().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        indices = target[valid] * self.num_classes + predicted[valid]
        counts = np.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        confusion = self.confusion.astype(np.float64)
        true_positive = np.diag(confusion)
        false_positive = confusion.sum(axis=0) - true_positive
        false_negative = confusion.sum(axis=1) - true_positive
        denominator = true_positive + false_positive + false_negative
        present = denominator > 0
        iou = np.divide(
            true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=present,
        )
        mean_iou = float(iou[present].mean()) if present.any() else 0.0
        total = confusion.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total > 0 else 0.0
        frequency = confusion.sum(axis=1) / total if total > 0 else np.zeros(self.num_classes)
        frequency_weighted_iou = (
            float((frequency[present] * iou[present]).sum() / frequency[present].sum())
            if present.any() and frequency[present].sum() > 0
            else 0.0
        )
        return {
            "miou": mean_iou,
            "pixel_accuracy": pixel_accuracy,
            "frequency_weighted_iou": frequency_weighted_iou,
            "per_class_iou": {
                CLASS_NAMES[index]: float(iou[index]) for index in range(self.num_classes)
            },
            "confusion_matrix": self.confusion.tolist(),
            "pixels": int(total),
        }
