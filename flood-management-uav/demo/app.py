"""Streamlit demo for FloodNet classification and segmentation checkpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

from floodnet.config import REPO_ROOT, load_config, resolve_device
from floodnet.data.transforms import (
    build_classification_transform,
    build_segmentation_transform,
)
from floodnet.engine.common import load_model_state
from floodnet.engine.predict import (
    _predict_classification_image,
    _predict_segmentation_image,
)
from floodnet.models.classifier import build_classifier
from floodnet.models.segmentor import build_segmentor
from floodnet.visualization import colorize_mask, legend


def _device(cfg: dict[str, Any]) -> torch.device:
    return torch.device(resolve_device(cfg))


@st.cache_resource(show_spinner=False)
def _load_classifier(checkpoint: str, config_path: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_config(config_path)
    model_cfg = cfg["classification"].get("model", {})
    model = build_classifier(
        arch=str(model_cfg.get("arch", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=False,
    )
    device = _device(cfg)
    load_model_state(Path(checkpoint), model, map_location=device)
    model.to(device).eval()
    return model, cfg


@st.cache_resource(show_spinner=False)
def _load_segmentor(checkpoint: str, config_path: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    cfg = load_config(config_path)
    model_cfg = cfg["segmentation"].get("model", {})
    model = build_segmentor(
        arch=str(model_cfg.get("arch", "deeplabv3plus")),
        encoder=str(model_cfg.get("encoder", "efficientnet-b3")),
        num_classes=int(model_cfg.get("num_classes", 10)),
        encoder_weights=None,
        encoder_depth=int(model_cfg.get("encoder_depth", 5)),
        decoder_channels=int(model_cfg.get("decoder_channels", 256)),
    )
    device = _device(cfg)
    load_model_state(Path(checkpoint), model, map_location=device)
    model.to(device).eval()
    return model, cfg


def _to_temp_file(data: bytes, suffix: str) -> Path:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(data)
    handle.close()
    return Path(handle.name)


def main() -> None:
    st.set_page_config(page_title="FloodNet", layout="wide")
    st.title("FloodNet UAV flood analysis")

    task = st.sidebar.selectbox("Task", ["Classification", "Segmentation"])
    default_checkpoint = (
        REPO_ROOT / "checkpoints" / ("cls_best.pt" if task == "Classification" else "seg_best.pt")
    )
    checkpoint = st.sidebar.text_input("Checkpoint", value=str(default_checkpoint))
    default_config = REPO_ROOT / "configs" / (
        "classification.yaml" if task == "Classification" else "segmentation.yaml"
    )
    config_path = st.sidebar.text_input("Config", value=str(default_config))
    uploaded = st.file_uploader("Upload a UAV image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded is None:
        st.info("Upload an image to run inference.")
        return
    if not Path(checkpoint).exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        return

    temp_path = _to_temp_file(uploaded.getvalue(), Path(uploaded.name).suffix or ".jpg")
    try:
        image_bgr = cv2.imread(str(temp_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not decode the uploaded image.")
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if task == "Classification":
            model, cfg = _load_classifier(checkpoint, config_path)
            image_size = cfg["data"]["classification"]["image_size"]
            transform = build_classification_transform(train=False)
            raw_prediction, probabilities = _predict_classification_image(
                model, temp_path, image_size, transform, _device(cfg)
            )
            invert = bool(cfg.get("legacy", {}).get("classification_output_inversion", True))
            prediction = raw_prediction ^ 1 if invert else raw_prediction
            if invert:
                probabilities = list(reversed(probabilities))
            label = "Flooded" if prediction == 1 else "Non-Flooded"

            left, right = st.columns([2, 1])
            with left:
                st.image(image_rgb, caption=uploaded.name, use_container_width=True)
            with right:
                st.metric("Prediction", label)
                st.write(
                    {
                        "Non-Flooded": f"{probabilities[0]:.2%}",
                        "Flooded": f"{probabilities[1]:.2%}",
                    }
                )
                st.caption(f"Raw class: {raw_prediction} | legacy inversion: {invert}")
        else:
            model, cfg = _load_segmentor(checkpoint, config_path)
            image_size = cfg["data"]["segmentation"]["image_size"]
            transform = build_segmentation_transform(
                image_size, train=False, legacy_val_uses_train=False
            )
            mask, _, _ = _predict_segmentation_image(
                model,
                temp_path,
                image_size,
                transform,
                _device(cfg),
                output_size=(image_bgr.shape[1], image_bgr.shape[0]),
            )
            color = colorize_mask(mask)
            overlay = cv2.addWeighted(
                cv2.resize(image_rgb, (mask.shape[1], mask.shape[0])),
                0.5,
                color,
                0.5,
                0.0,
            )
            flood_pixels = int(np.isin(mask, [1, 3, 5]).sum())
            flood_fraction = flood_pixels / mask.size if mask.size else 0.0
            left, right = st.columns(2)
            with left:
                st.image(image_rgb, caption="Input", use_container_width=True)
            with right:
                st.image(overlay, caption="Prediction overlay", use_container_width=True)
            st.metric("Flooded area", f"{flood_fraction:.2%}")
            st.dataframe(legend(), use_container_width=True)
            success, encoded = cv2.imencode(".png", mask)
            if success:
                st.download_button(
                    "Download indexed mask",
                    data=encoded.tobytes(),
                    file_name=f"{Path(uploaded.name).stem}_mask.png",
                    mime="image/png",
                )
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
