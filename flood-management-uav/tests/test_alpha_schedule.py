from floodnet.semi_supervised import classification_alpha, segmentation_alpha


def test_classification_alpha_matches_notebook() -> None:
    assert classification_alpha(0) == 0.0
    assert classification_alpha(9) == 0.0
    assert classification_alpha(10) == 0.0
    assert classification_alpha(11) == 1.0 / 139
    assert classification_alpha(149) == 1.0
    assert classification_alpha(150) == 1.0
    assert classification_alpha(500) == 1.0


def test_segmentation_alpha_matches_notebook() -> None:
    assert segmentation_alpha(0) == 0.0
    assert segmentation_alpha(14) == 0.0
    assert segmentation_alpha(15) == 0.0
    assert abs(segmentation_alpha(16) - (0.5 / 639)) < 1e-12
    assert abs(segmentation_alpha(99) - 42.0 / 639) < 1e-12
    assert segmentation_alpha(655) == 0.5
    assert segmentation_alpha(1000) == 0.5
