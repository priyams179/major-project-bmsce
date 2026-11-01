import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from floodnet.engine.train_classifier import train_classifier
from floodnet.engine.train_segmentor import train_segmentor


def _image(path: Path, value: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((64, 64, 3), value, dtype=np.uint8))


def _mask(path: Path, value: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32:, :] = value
    cv2.imwrite(str(path), mask)


def _item(root: Path, name: str, label: int, mask: bool) -> dict[str, object]:
    image_path = root / f"{name}.jpg"
    _image(image_path, value=80 + label * 40)
    payload: dict[str, object] = {
        "name": name,
        "image": str(image_path),
        "label": label,
        "source": "synthetic",
    }
    if mask:
        mask_path = root / f"{name}_mask.png"
        _mask(mask_path, value=1 if label else 5)
        payload["mask"] = str(mask_path)
    else:
        payload["mask"] = None
    return payload


def test_training_smoke_cpu(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    cls_root = root / "classification"
    seg_root = root / "segmentation"
    cls_root.mkdir(parents=True)
    seg_root.mkdir(parents=True)

    cls_splits = {
        "train": [_item(root, "c_train_0", 0, False), _item(root, "c_train_1", 1, False)],
        "val": [_item(root, "c_val_0", 0, False), _item(root, "c_val_1", 1, False)],
        "unlabeled": [_item(root, "c_unlab_0", -1, False), _item(root, "c_unlab_1", -1, False)],
        "official_val": [],
        "official_test": [],
    }
    (cls_root / "manifest.json").write_text(
        json.dumps(
            {
                "task": "classification",
                "image_size": [64, 64],
                "num_classes": 2,
                "splits": cls_splits,
            }
        ),
        encoding="utf-8",
    )

    seg_splits = {
        "train": [_item(root, "s_train_0", 0, True), _item(root, "s_train_1", 1, True)],
        "val": [_item(root, "s_val_0", 0, True), _item(root, "s_val_1", 1, True)],
        "unlabeled": [_item(root, "s_unlab_0", -1, False), _item(root, "s_unlab_1", -1, False)],
    }
    (seg_root / "manifest.json").write_text(
        json.dumps(
            {
                "task": "segmentation",
                "image_size": [64, 64],
                "num_classes": 10,
                "splits": seg_splits,
            }
        ),
        encoding="utf-8",
    )

    config = {
        "seed": 7,
        "device": "cpu",
        "paths": {
            "raw_root": str(tmp_path / "raw" / "floodnet"),
            "processed_root": str(root),
            "checkpoints": str(tmp_path / "checkpoints"),
            "outputs": str(tmp_path / "outputs"),
        },
        "data": {
            "num_workers": 0,
            "classification": {"image_size": [64, 64]},
            "segmentation": {
                "image_size": [64, 64],
                "unlabeled_limit": 2,
                "val_uses_train_transform": True,
            },
        },
        "classification": {
            "train": {"epochs": 1, "batch_size": 2, "num_workers": 0},
            "output_dir": str(tmp_path / "checkpoints"),
            "checkpoint_name": "cls_test.pt",
        },
        "segmentation": {
            "train": {
                "supervised_epochs": 1,
                "semi_supervised_epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "save_every": 0,
            },
            "output_dir": str(tmp_path / "checkpoints"),
            "checkpoint_name": "seg_test.pt",
        },
        "logging": {"tensorboard": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    classifier_args = argparse.Namespace(
        config=config_path,
        epochs=1,
        batch_size=2,
        device="cpu",
        smoke=True,
        no_pretrained=True,
    )
    classifier_checkpoint = train_classifier(classifier_args)
    assert classifier_checkpoint.exists()

    segmentor_args = argparse.Namespace(
        config=config_path,
        stage="both",
        supervised_epochs=1,
        semi_supervised_epochs=1,
        batch_size=2,
        device="cpu",
        resume=None,
        smoke=True,
        no_pretrained=True,
    )
    segmentor_checkpoint = train_segmentor(segmentor_args)
    assert segmentor_checkpoint.exists()

    from floodnet.engine.evaluate import evaluate
    from floodnet.engine.predict import predict

    classification_eval = argparse.Namespace(
        task="classification",
        split="val",
        checkpoint=classifier_checkpoint,
        config=config_path,
        output=tmp_path / "outputs" / "cls_eval.json",
        device="cpu",
        batch_size=2,
        legacy_invert=False,
    )
    cls_metrics = evaluate(classification_eval)
    assert classification_eval.output.exists()
    assert cls_metrics["metrics"] is not None

    classification_prediction = argparse.Namespace(
        task="classification",
        input=root / "c_val_0.jpg",
        output=tmp_path / "outputs" / "cls_predict",
        checkpoint=classifier_checkpoint,
        config=config_path,
        device="cpu",
        legacy_invert=False,
        output_size=None,
    )
    predict(classification_prediction)
    assert (classification_prediction.output / "image_classes.json").exists()

    segmentation_eval = argparse.Namespace(
        task="segmentation",
        split="val",
        checkpoint=segmentor_checkpoint,
        config=config_path,
        output=tmp_path / "outputs" / "seg_eval",
        device="cpu",
        batch_size=2,
        legacy_invert=False,
    )
    seg_metrics = evaluate(segmentation_eval)
    assert seg_metrics["num_items"] == 2
    assert Path(seg_metrics["predictions_dir"]).exists()

    segmentation_prediction = argparse.Namespace(
        task="segmentation",
        input=root / "s_val_0.jpg",
        output=tmp_path / "outputs" / "seg_predict",
        checkpoint=segmentor_checkpoint,
        config=config_path,
        device="cpu",
        legacy_invert=False,
        output_size="64x64",
    )
    predict(segmentation_prediction)
    assert (segmentation_prediction.output / "masks" / "s_val_0.png").exists()
