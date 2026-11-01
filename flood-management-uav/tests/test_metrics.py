import torch

from floodnet.metrics import ClassificationMetrics, SegmentationMetrics


def test_classification_metrics_perfect() -> None:
    metrics = ClassificationMetrics()
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 0])
    metrics.update(logits, labels)
    result = metrics.compute()
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["samples"] == 4
    assert result["confusion_matrix"] == [[2, 0], [0, 2]]


def test_classification_metrics_mixed() -> None:
    metrics = ClassificationMetrics()
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 1, 0])
    metrics.update(logits, labels)
    result = metrics.compute()
    assert result["accuracy"] == 0.5
    assert 0.0 < result["macro_f1"] < 1.0


def test_segmentation_metrics_perfect() -> None:
    metrics = SegmentationMetrics(num_classes=3)
    target = torch.tensor([[0, 1], [2, 2]])
    metrics.update(target, target)
    result = metrics.compute()
    assert result["miou"] == 1.0
    assert result["pixel_accuracy"] == 1.0
    assert result["per_class_iou"]["Background"] == 1.0


def test_segmentation_metrics_partial() -> None:
    metrics = SegmentationMetrics(num_classes=2)
    target = torch.tensor([[0, 0], [0, 0]])
    prediction = torch.tensor([[0, 0], [0, 1]])
    metrics.update(prediction, target)
    result = metrics.compute()
    assert result["pixels"] == 4
    assert result["pixel_accuracy"] == 0.75
    assert result["per_class_iou"]["Background"] < 1.0
