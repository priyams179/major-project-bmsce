"""Shared constants for FloodNet classification and segmentation."""

from __future__ import annotations

NUM_CLASSES = 10

CLASS_NAMES: dict[int, str] = {
    0: "Background",
    1: "Building Flooded",
    2: "Building Non-Flooded",
    3: "Road Flooded",
    4: "Road Non-Flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass",
}

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (200, 90, 90),
    3: (128, 128, 0),
    4: (155, 155, 155),
    5: (0, 255, 255),
    6: (55, 0, 255),
    7: (255, 0, 255),
    8: (245, 245, 0),
    9: (0, 255, 0),
}

FLOOD_CLASSES: tuple[int, ...] = (1, 3, 5)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASSIFICATION_LABELS: dict[int, str] = {0: "Non-Flooded", 1: "Flooded"}

IGNORE_INDEX = 255
SEGMENTATION_OUTPUT_SIZE = (4000, 3000)

HF_REPO_ID = "torchgeo/floodnet"
HF_FILENAME = "floodnet.zip"
HF_SHA256 = "962e6dac95495d7fdc249681d3973889f7e93487fb9e3da25e1df4f451f4f272"
DROPBOX_URL = (
    "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/"
    "ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&e=2&dl=0"
)

EXPECTED_LABELED_FLOODED = 51
EXPECTED_LABELED_NON_FLOODED = 347
EXPECTED_UNLABELED = 1047
EXPECTED_VALIDATION = 450
EXPECTED_TEST = 448
