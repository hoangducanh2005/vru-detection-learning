"""Các helper nhỏ dùng chung cho executable scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resized_bgr(path: str, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    interpolation = cv2.INTER_AREA if size[0] < image.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def load_model_state(model: torch.nn.Module, path: str | Path, strict: bool = True) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    result = model.load_state_dict(state, strict=strict)
    if not strict:
        print(f"checkpoint load: missing={list(result.missing_keys)}, unexpected={list(result.unexpected_keys)}")
    return checkpoint if isinstance(checkpoint, dict) else {}
