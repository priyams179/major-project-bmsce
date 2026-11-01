"""Alpha schedules and pseudo-label helpers."""

from __future__ import annotations

import numpy as np


def classification_alpha(
    epoch: int,
    start: int = 10,
    end: int = 150,
    max_alpha: float = 1.0,
) -> float:
    """Exact notebook T1 schedule."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return float(max_alpha)
    alphas = np.linspace(0.0, float(max_alpha), end - start)
    return float(alphas[epoch - start])


def segmentation_alpha(
    epoch: int,
    start: int = 15,
    end: int = 655,
    max_alpha: float = 0.5,
) -> float:
    """Exact notebook T2 schedule.

    With a 100-epoch semi-supervised run the maximum reached alpha is about
    0.0657, because the notebook ramps through epoch 655.
    """
    if epoch < start:
        return 0.0
    alphas = np.linspace(0.0, float(max_alpha), end - start)
    index = epoch - start
    if index >= len(alphas):
        return float(alphas[-1])
    return float(alphas[index])
