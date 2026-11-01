"""Configuration loading and CLI helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    base_path: str | Path | None = DEFAULT_BASE_CONFIG,
) -> dict[str, Any]:
    """Load a task config on top of base.yaml.

    Paths in the YAML are interpreted relative to the repository root so the
    CLI behaves the same when invoked from any directory.
    """
    base = _read_yaml(Path(base_path)) if base_path is not None else {}
    override = _read_yaml(Path(path)) if path is not None else {}
    cfg = deep_merge(base, override)
    _resolve_config_paths(cfg)
    return cfg


def _resolve_config_paths(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return
    for key, value in list(paths.items()):
        if isinstance(value, str) and value and not Path(value).is_absolute():
            paths[key] = str(REPO_ROOT / value)


def resolve_device(cfg: dict[str, Any] | None = None) -> str:
    import torch

    requested = "auto"
    if cfg is not None:
        requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def get_path(cfg: dict[str, Any], key: str) -> Path:
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Missing paths.{key} in config")
    return Path(paths[key])


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    for key in ("processed_root", "checkpoints", "outputs"):
        get_path(cfg, key).mkdir(parents=True, exist_ok=True)
