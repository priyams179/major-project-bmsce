import torch

from floodnet.losses import BCEDiceLoss, dice_loss


def test_bce_dice_loss_is_half_bce_half_dice() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 4)
    target = torch.rand(2, 3, 4, 4)
    criterion = BCEDiceLoss(bce_weight=0.5, smooth=1e-5)
    actual = criterion(logits, target)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(0, 2, 3))
    cardinality = probabilities.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1e-5) / (cardinality + 1e-5)).mean()
    expected = 0.5 * bce + 0.5 * dice
    assert torch.allclose(actual, expected, atol=1e-6)


def test_bce_dice_accepts_legacy_raw_logit_targets() -> None:
    torch.manual_seed(1)
    logits = torch.randn(1, 2, 4, 4)
    pseudo_logits = torch.randn(1, 2, 4, 4)
    loss = BCEDiceLoss()(logits, pseudo_logits)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_dice_loss_zero_for_perfect_mask() -> None:
    target = torch.ones(2, 3, 4, 4)
    probabilities = torch.ones_like(target)
    assert dice_loss(probabilities, target).item() < 1e-6
