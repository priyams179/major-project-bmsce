"""
Loss functions for landslide segmentation from multi-source satellite imagery.

Ported and extended from the flood-management pipeline (cnn_model.py).

Landslide pixels are typically <1% of the scene, causing severe class imbalance.
All losses here operate only on *valid* pixels, defined by a ``valid_mask`` tensor
(float32) where 1.0 = observed / valid, 0.0 = cloud / nodata / out-of-scene.
Pixels masked out (target == -1) are excluded from every gradient computation.

Available loss classes
----------------------
MaskedDiceLoss          – Soft Dice over valid pixels only.
MaskedBCEDiceLoss       – 0.5 * BCE + 0.5 * Dice with mask-awareness.
MaskedFocalTverskyLoss  – Focal Tversky for extreme class imbalance.
CombinedLoss            – Configurable weighted mix of FocalTversky + Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: extract valid (non-masked) pixels from flat tensors
# ---------------------------------------------------------------------------
def _apply_mask(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None):
    """
    Flatten probs and targets and, if *mask* is provided, keep only valid pixels.

    Parameters
    ----------
    probs   : torch.Tensor – sigmoid probabilities, any shape.
    targets : torch.Tensor – binary ground-truth, same shape as *probs*.
    mask    : torch.Tensor or None – float mask (1=valid, 0=invalid), same shape.

    Returns
    -------
    probs_flat, targets_flat : 1-D tensors of valid pixels.
    is_empty                 : bool – True when no valid pixel exists.
    """
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)

    if mask is not None:
        mask_flat = mask.view(-1)
        valid_idx = mask_flat > 0.5
        if not valid_idx.any():
            return probs_flat, targets_flat, True   # signal "no valid pixels"
        probs_flat = probs_flat[valid_idx]
        targets_flat = targets_flat[valid_idx]

    return probs_flat, targets_flat, False


# ---------------------------------------------------------------------------
# 1. MaskedDiceLoss
# ---------------------------------------------------------------------------
class MaskedDiceLoss(nn.Module):
    """
    Soft Dice Loss computed strictly over valid mask pixels.

    The Dice coefficient measures the overlap between predicted probabilities
    and binary ground-truth labels. Because landslide pixels are rare, Dice
    loss naturally up-weights the minority class without requiring explicit
    class weights.

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to avoid division by zero (default 1.0).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] – raw (pre-sigmoid) network outputs.
        targets : Tensor [B, 1, H, W] – binary ground-truth (0 or 1).
        mask    : Tensor [B, 1, H, W] or None – float validity mask.

        Returns
        -------
        Scalar loss tensor.
        """
        probs = torch.sigmoid(logits)
        probs_flat, targets_flat, is_empty = _apply_mask(probs, targets, mask)

        if is_empty:
            # No valid pixels – return differentiable zero
            return logits.sum() * 0.0

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


# ---------------------------------------------------------------------------
# 2. MaskedBCEDiceLoss
# ---------------------------------------------------------------------------
class MaskedBCEDiceLoss(nn.Module):
    """
    Combined Masked Binary Cross-Entropy + Masked Soft Dice Loss.

    Loss = bce_weight * BCE  +  dice_weight * Dice

    By default each component has weight 0.5 (equal contribution). BCE
    provides stable gradients at every pixel; Dice corrects for class
    imbalance by optimising the overlap ratio. Together they outperform
    either loss alone on highly imbalanced segmentation benchmarks.

    Parameters
    ----------
    bce_weight  : float – Weight on the BCE component (default 0.5).
    dice_weight : float – Weight on the Dice component (default 0.5).
    pos_weight  : float or None – Optional scalar multiplier on the positive
                  class in BCE (further rebalancing, e.g. 5.0 for 1:5 ratio).
    smooth      : float – Dice smoothing constant (default 1.0).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.dice = MaskedDiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] – raw network outputs.
        targets : Tensor [B, 1, H, W] – binary ground-truth.
        mask    : Tensor [B, 1, H, W] or None – float validity mask.

        Returns
        -------
        Scalar loss tensor.
        """
        # Move pos_weight to the correct device lazily
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)

        # --- BCE component (element-wise, then masked mean) ---
        bce_raw = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if mask is not None:
            valid_pixels = mask.sum()
            bce_loss = (
                (bce_raw * mask).sum() / valid_pixels
                if valid_pixels > 0
                else logits.sum() * 0.0
            )
        else:
            bce_loss = bce_raw.mean()

        # --- Dice component ---
        dice_loss = self.dice(logits, targets, mask=mask)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# 3. MaskedFocalTverskyLoss
# ---------------------------------------------------------------------------
class MaskedFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for extreme foreground / background class imbalance.

    Tversky Index
    -------------
    The Tversky index generalises the Dice coefficient by independently
    weighting false positives (FP) and false negatives (FN):

        TI = TP / (TP + alpha * FP + beta * FN + smooth)

    By setting  alpha < 0.5  and  beta > 0.5  we penalise false negatives
    (missed landslides) more than false positives.  This is crucial when
    landslide pixels are < 1 % of the scene, because a naive model achieves
    > 99 % pixel accuracy by predicting "no landslide" everywhere, yet it
    misses all real events.

    alpha  controls the FP penalty: high alpha → higher precision (fewer FP).
    beta   controls the FN penalty: high beta  → higher recall   (fewer FN).
    Note: alpha + beta need not equal 1 but conventionally do.

    Focal Modification
    ------------------
    The focal parameter *gamma* applies an adaptive difficulty weighting,
    identical in spirit to the focal loss for object detection [Lin et al.]:

        L = (1 - TI)^gamma

    When gamma > 1, easy-to-classify examples (high TI) contribute less to
    the gradient, focusing training on hard, under-represented landslide
    pixels.

    Default Hyperparameters
    -----------------------
    alpha = 0.3  – low FP penalty (accept some false positives to find all
                   landslides; recall is more operationally critical).
    beta  = 0.7  – high FN penalty (minimise missed detections).
    gamma = 1.33 – mild focal sharpening; empirically effective on small
                   footprint landslide datasets.

    Parameters
    ----------
    alpha  : float – FP weight in Tversky denominator (default 0.3).
    beta   : float – FN weight in Tversky denominator (default 0.7).
    gamma  : float – Focal exponent; higher = harder example focus (default 1.33).
    smooth : float – Laplace smoothing to prevent division by zero (default 1.0).

    References
    ----------
    Salehi et al. (2017) "Tversky loss function for image segmentation
    using 3D fully convolutional deep networks." MLMI 2017.
    Abraham & Khan (2019) "A novel focal tversky loss function with improved
    attention u-net for lesion segmentation." ISBI 2019.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha   # FP weight – low → tolerate FP to boost recall
        self.beta = beta     # FN weight – high → heavily penalise missed events
        self.gamma = gamma   # focal exponent
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] – raw network outputs.
        targets : Tensor [B, 1, H, W] – binary ground-truth.
        mask    : Tensor [B, 1, H, W] or None – float validity mask.

        Returns
        -------
        Scalar focal Tversky loss.
        """
        probs = torch.sigmoid(logits)
        probs_flat, targets_flat, is_empty = _apply_mask(probs, targets, mask)

        if is_empty:
            return logits.sum() * 0.0

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky_idx = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_tversky = torch.pow(1.0 - tversky_idx, self.gamma)
        return focal_tversky


# ---------------------------------------------------------------------------
# 4. CombinedLoss
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Configurable weighted combination of Focal Tversky Loss + Soft Dice Loss.

    This is the recommended training loss for the landslide segmentation
    pipeline. The Focal Tversky component addresses the extreme class
    imbalance, while the Dice component stabilises training by providing
    dense gradient signal over the entire valid region.

    Loss = ft_weight * FocalTversky + dice_weight * Dice

    Parameters
    ----------
    ft_weight   : float – Weight on the Focal Tversky component (default 0.7).
    dice_weight : float – Weight on the Dice component (default 0.3).
    alpha       : float – Tversky FP weight (passed to MaskedFocalTverskyLoss).
    beta        : float – Tversky FN weight (passed to MaskedFocalTverskyLoss).
    gamma       : float – Focal exponent (passed to MaskedFocalTverskyLoss).
    smooth      : float – Dice / Tversky smoothing constant.
    """

    def __init__(
        self,
        ft_weight: float = 0.7,
        dice_weight: float = 0.3,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ft_weight = ft_weight
        self.dice_weight = dice_weight
        self.focal_tversky = MaskedFocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma, smooth=smooth
        )
        self.dice = MaskedDiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] – raw network outputs.
        targets : Tensor [B, 1, H, W] – binary ground-truth.
        mask    : Tensor [B, 1, H, W] or None – float validity mask.

        Returns
        -------
        Scalar combined loss.
        """
        ft_loss = self.focal_tversky(logits, targets, mask=mask)
        dice_loss = self.dice(logits, targets, mask=mask)
        return self.ft_weight * ft_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# 5. Lovasz-hinge losses
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension w.r.t. sorted errors (Berman et al.,
    CVPR 2018).  `gt_sorted` is the ground-truth flags sorted by model error,
    descending."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class MaskedLovaszHingeLoss(nn.Module):
    """
    Lovasz-hinge loss over valid pixels (binary segmentation).

    Directly optimises the Jaccard index (IoU) via its Lovasz extension, so
    the training objective matches the evaluation metric instead of proxying
    it through per-pixel losses.  In the Landslide4Sense 2022 winning
    solutions this loss was worth roughly +2.5 F1 over plain BCE-style
    training (2nd-place team ablation).

    Parameters
    ----------
    per_image : bool – Compute the loss per patch and average (True, the
        default, matches the reference implementation) or over the whole
        batch at once.
    """

    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, 1, H, W] – raw network outputs.
        targets : Tensor [B, 1, H, W] – binary ground-truth.
        mask    : Tensor [B, 1, H, W] or None – float validity mask; pixels
            where mask == 0 are excluded from the loss.
        """
        losses = []
        for b in range(logits.shape[0]):
            logit = logits[b].reshape(-1)
            tgt = (targets[b].reshape(-1) > 0.5).float()
            if mask is not None:
                valid = mask[b].reshape(-1) > 0.5
                logit, tgt = logit[valid], tgt[valid]
            if tgt.numel() == 0:
                continue
            errors = torch.abs(tgt - torch.sigmoid(logit))
            errors_sorted, perm = torch.sort(errors, descending=True)
            tgt_sorted = tgt[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(tgt_sorted)))
        if not losses:
            return logits.sum() * 0.0  # graph-connected zero
        return sum(losses) / len(losses)


class MaskedBCELovaszLoss(nn.Module):
    """
    50/50 blend of masked BCE and Lovasz-hinge.

    The BCE term supplies dense per-pixel gradients early in training while
    the Lovasz term aligns the objective with IoU (the combination used by
    the 2nd-place L4S 2022 team, in their case weighted cross-entropy +
    Lovasz).
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = 1.0 - bce_weight
        self.bce = MaskedBCEDiceLoss(bce_weight=1.0, dice_weight=0.0,
                                     pos_weight=pos_weight)
        self.lovasz = MaskedLovaszHingeLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted sum of masked BCE and masked Lovasz-hinge."""
        return (self.bce_weight * self.bce(logits, targets, mask=mask)
                + self.lovasz_weight * self.lovasz(logits, targets, mask=mask))
