"""
Landslide detection models package.

Public API
----------
Segmentation
    get_model           – factory for UNet / ResNet34UNet.

Change detection
    get_cd_model        – factory for SNUNet-CD.

Loss functions
    MaskedFocalTverskyLoss  – primary loss for extreme class imbalance.
    MaskedBCEDiceLoss       – combined BCE + Dice loss.
    MaskedDiceLoss          – soft Dice over valid pixels.
    CombinedLoss            – configurable FocalTversky + Dice blend.

Metrics
    ConfusionMatrixTracker  – stateful dataset-level metric accumulator.
    compute_metrics         – stateless function from raw TP/FP/FN/TN counts.
    optimal_threshold       – threshold search maximising F1.
"""

from .segmentation_model import get_model                                       # noqa: F401
from .change_detection_model import get_cd_model                                # noqa: F401
from .losses import (                                                           # noqa: F401
    MaskedFocalTverskyLoss,
    MaskedBCEDiceLoss,
    MaskedDiceLoss,
    CombinedLoss,
)
from .metrics import (                                                          # noqa: F401
    ConfusionMatrixTracker,
    compute_metrics,
    optimal_threshold,
)

__all__ = [
    # Segmentation
    "get_model",
    # Change detection
    "get_cd_model",
    # Losses
    "MaskedFocalTverskyLoss",
    "MaskedBCEDiceLoss",
    "MaskedDiceLoss",
    "CombinedLoss",
    # Metrics
    "ConfusionMatrixTracker",
    "compute_metrics",
    "optimal_threshold",
]
