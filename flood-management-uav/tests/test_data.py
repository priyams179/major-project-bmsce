from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from floodnet.data.classification_dataset import ClassificationDataset
from floodnet.data.segmentation_dataset import SegmentationDataset, one_hot_masks
from floodnet.data.transforms import (
    build_classification_transform,
    build_segmentation_transform,
)
from floodnet.data.validate import find_mask_for_image, validate_dataset
from floodnet.engine.train_classifier import _generate_pseudo_labels


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[12:, :] = 5
    cv2.imwrite(str(path), mask)


def test_find_mask_and_validation_report(tmp_path: Path) -> None:
    raw = tmp_path / "floodnet"
    _write_image(raw / "Train/Labeled/Flooded/image/a.jpg")
    _write_mask(raw / "Train/Labeled/Flooded/mask/a_lab.png")
    _write_image(raw / "Train/Labeled/Non-Flooded/image/b.jpg")
    _write_image(raw / "Train/Unlabeled/image/c.jpg")
    _write_image(raw / "Validation/image/d.jpg")
    _write_image(raw / "Test/image/e.jpg")

    assert find_mask_for_image(
        raw / "Train/Labeled/Flooded/image/a.jpg",
        raw / "Train/Labeled/Flooded/mask",
    ) == raw / "Train/Labeled/Flooded/mask/a_lab.png"

    summary = validate_dataset(raw, validate_masks=True, strict=False)
    assert summary["pairs"] == 1
    assert any("labeled_flooded" in issue for issue in summary["issues"])


def test_segmentation_dataset_returns_tensor_and_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    mask_path = tmp_path / "mask.png"
    _write_image(image_path)
    _write_mask(mask_path)
    items = [{"name": "sample", "image": str(image_path), "mask": str(mask_path)}]
    dataset = SegmentationDataset(
        items,
        transform=build_segmentation_transform([16, 16], train=True),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["mask"].shape == (16, 16)
    assert sample["mask"].max() < 10
    one_hot = one_hot_masks(sample["mask"].unsqueeze(0))
    assert one_hot.shape == (1, 10, 16, 16)
    assert torch.allclose(one_hot.sum(dim=1), torch.ones(1, 16, 16))


def test_unlabeled_segmentation_dataset_has_no_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    _write_image(image_path)
    dataset = SegmentationDataset(
        [{"name": "sample", "image": str(image_path), "mask": None}],
        transform=build_segmentation_transform([16, 16], train=True),
        unlabeled=True,
    )
    sample = dataset[0]
    assert "mask" not in sample
    assert sample["image"].shape == (3, 16, 16)


def test_classification_dataset(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    _write_image(image_path, value=200)
    dataset = ClassificationDataset(
        [{"name": "sample", "image": str(image_path), "label": 1}],
        transform=build_classification_transform(train=False),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 24, 24)
    assert sample["label"] == 1
    assert sample["index"] == 0


class _IndexDataset(Dataset):
    def __len__(self) -> int:
        return 7

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"image": torch.tensor([float(index)]), "index": index}


class _IndexModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        logits = torch.zeros(batch, 2, 1, 1)
        logits[:, 1, 0, 0] = images[:, 0] % 2
        return logits


def test_classification_pseudo_labels_use_dataset_index() -> None:
    dataset = _IndexDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    pseudo = _generate_pseudo_labels(_IndexModel(), loader, torch.device("cpu"), len(dataset))
    expected = torch.arange(len(dataset)) % 2
    assert torch.equal(pseudo, expected.long())


def test_download_extracts_and_normalizes_dataset_root(tmp_path: Path) -> None:
    import zipfile

    from floodnet.data.download import download_dataset

    archive = tmp_path / "manual.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("FloodNet Track 1/Train/Labeled/Flooded/image/a.jpg", b"image")
        handle.writestr("FloodNet Track 1/Test/image/b.jpg", b"image")

    dest = tmp_path / "raw"
    root = download_dataset(dest=dest, zip_path=archive, verify=False)
    assert root == dest.resolve() / "floodnet"
    assert (root / "Train/Labeled/Flooded/image/a.jpg").exists()
    assert not (dest / "FloodNet Track 1").exists()


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    import zipfile

    import pytest

    from floodnet.data.download import safe_extract

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", b"bad")
    with pytest.raises(ValueError, match="path traversal"):
        safe_extract(archive, tmp_path / "out")
