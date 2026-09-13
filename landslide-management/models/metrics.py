"""
Confusion-matrix metrics for landslide segmentation evaluation.

Ported and extended from the flood-management pipeline (cnn_model.py).

Key design choice: metrics are accumulated at the *dataset level* (global TP/FP/FN/TN)
rather than averaged per batch.  This avoids the "batch mean" bias that arises
when batch sizes or the number of valid pixels per batch vary—an important
consideration for tiles that contain many cloud/nodata pixels.

Available
---------
ConfusionMatrixTracker  – Stateful class; accumulate across batches then compute.
compute_metrics         – Stateless function; compute metrics from raw counts.
optimal_threshold       – Scan [0.1, 0.9] to find the threshold maximising F1.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Stateless helper
# ---------------------------------------------------------------------------
def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """
    Compute segmentation quality metrics from raw confusion-matrix counts.

    Parameters
    ----------
    tp, fp, fn, tn : int
        True positives, false positives, false negatives, true negatives
        accumulated over the evaluation set.

    Returns
    -------
    dict with keys:
        iou        – Intersection-over-Union (Jaccard index) for the positive class.
        f1         – F1 / Dice score for the positive class.
        precision  – Positive predictive value.
        recall     – Sensitivity / true positive rate.
        accuracy   – Overall pixel accuracy (valid pixels only).
        tp, fp, fn, tn – raw counts (passed through).
    """
    total_pred_pos = tp + fp
    total_pos = tp + fn
    total_valid = tp + fp + fn + tn
    union = tp + fp + fn

    precision = tp / total_pred_pos if total_pred_pos > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / union if union > 0 else 0.0
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0.0

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ---------------------------------------------------------------------------
# Stateful tracker
# ---------------------------------------------------------------------------
class ConfusionMatrixTracker:
    """
    Dataset-level confusion matrix accumulator for binary segmentation.

    Collects global TP, FP, FN, TN across many batches, then computes
    exact dataset-level metrics in a single :py:meth:`compute` call.

    This avoids the micro-averaging artefact that occurs when computing
    metrics per batch and then averaging, particularly when batches have
    different numbers of valid pixels (due to clouds or nodata).

    Parameters
    ----------
    threshold : float
        Sigmoid threshold used to binarise network outputs (default 0.5).

    Example
    -------
    >>> tracker = ConfusionMatrixTracker(threshold=0.5)
    >>> for logits, targets, mask in val_loader:
    ...     tracker.update(logits, targets, mask)
    >>> metrics = tracker.compute()
    >>> print(f"IoU = {metrics['iou']:.4f},  F1 = {metrics['f1']:.4f}")
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all accumulators (call between epochs / evaluation runs)."""
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

    # ------------------------------------------------------------------
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Accumulate counts for one mini-batch.

        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] or [B, H, W]
            Raw (pre-sigmoid) network outputs.
        targets : Tensor – same shape as *logits*
            Binary ground-truth labels (0 or 1).
        mask    : Tensor or None – float validity mask (1=valid, 0=cloud/nodata).
            When provided, only pixels where mask > 0.5 are counted.
        """
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > self.threshold).bool()
            tgts = (targets > 0.5).bool()

            if mask is not None:
                valid = (mask > 0.5).bool()
                # Flatten boolean index across all spatial dims
                preds = preds[valid]
                tgts = tgts[valid]
            else:
                preds = preds.reshape(-1)
                tgts = tgts.reshape(-1)

            self.tp += int((preds & tgts).sum().item())
            self.fp += int((preds & ~tgts).sum().item())
            self.fn += int((~preds & tgts).sum().item())
            self.tn += int((~preds & ~tgts).sum().item())

    # ------------------------------------------------------------------
    def compute(self) -> dict[str, float | int]:
        """
        Compute all metrics from accumulated counts.

        Returns
        -------
        dict with keys: iou, f1, precision, recall, accuracy, tp, fp, fn, tn.
        """
        return compute_metrics(self.tp, self.fp, self.fn, self.tn)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        m = self.compute()
        return (
            f"ConfusionMatrixTracker("
            f"IoU={m['iou']:.4f}, F1={m['f1']:.4f}, "
            f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})"
        )


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------
def optimal_threshold(
    logits_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    mask_list: Optional[list[torch.Tensor]] = None,
    num_steps: int = 17,
) -> tuple[float, dict[str, float | int]]:
    """
    Find the sigmoid threshold that maximises the dataset-level F1 score.

    Scans *num_steps* evenly-spaced thresholds in [0.1, 0.9] and returns
    the threshold with the highest F1.  All tensors are expected to already
    be on CPU (or the same device) and are not moved internally.

    Parameters
    ----------
    logits_list  : list of Tensor – raw (pre-sigmoid) outputs, one per batch.
    targets_list : list of Tensor – binary ground-truth, matching *logits_list*.
    mask_list    : list of Tensor or None – validity masks, matching *logits_list*.
                   Pass ``None`` for the entire list to disable masking.
    num_steps    : int – number of threshold candidates to evaluate (default 17).

    Returns
    -------
    best_threshold : float – threshold in [0.1, 0.9] maximising F1.
    best_metrics   : dict  – metrics dict at the best threshold.

    Example
    -------
    >>> thresh, metrics = optimal_threshold(all_logits, all_targets, all_masks)
    >>> print(f"Best threshold: {thresh:.2f}  →  F1 = {metrics['f1']:.4f}")
    """
    if mask_list is None:
        mask_list = [None] * len(logits_list)

    if len(logits_list) != len(targets_list) or len(logits_list) != len(mask_list):
        raise ValueError(
            "logits_list, targets_list, and mask_list must have the same length."
        )

    thresholds = np.linspace(0.1, 0.9, num_steps)
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics: dict = {}

    for thresh in thresholds:
        tracker = ConfusionMatrixTracker(threshold=float(thresh))
        for logits, targets, mask in zip(logits_list, targets_list, mask_list):
            tracker.update(logits, targets, mask=mask)
        metrics = tracker.compute()
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(thresh)
            best_metrics = metrics

    return best_threshold, best_metrics
