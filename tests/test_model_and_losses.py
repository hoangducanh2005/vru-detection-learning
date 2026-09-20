"""Test nhanh tensor contract, không phụ thuộc accuracy của model."""

import pytest
import torch

from src.losses.depth_loss import sparse_depth_loss
from src.losses.detection_loss import detection_loss
from src.model.vru_model import VruModel
from src.utils.decode import decode_detections


@pytest.fixture(scope="module")
def model_outputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = VruModel(fpn_channels=32, with_depth=True, depth_bins=16)
    model.eval()
    image = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        return model(image)


def test_detector_forward_shapes(model_outputs: dict[str, torch.Tensor]) -> None:
    outputs = model_outputs
    assert outputs["feature"].shape == (1, 32, 16, 24)
    assert outputs["heatmap"].shape == (1, 2, 16, 24)
    assert outputs["offset"].shape == (1, 2, 16, 24)
    assert outputs["size"].shape == (1, 2, 16, 24)


def test_depth_head_forward_shape(model_outputs: dict[str, torch.Tensor]) -> None:
    outputs = model_outputs
    assert outputs["depth_logits"].shape == (1, 16, 16, 24)


def test_detection_loss_is_finite(model_outputs: dict[str, torch.Tensor]) -> None:
    outputs = model_outputs
    targets = {
        "heatmap": torch.zeros_like(outputs["heatmap"]),
        "offset": torch.zeros_like(outputs["offset"]),
        "size": torch.zeros_like(outputs["size"]),
        "mask": torch.zeros(1, 1, 16, 24),
    }
    targets["heatmap"][0, 1, 8, 12] = 1.0
    targets["mask"][0, 0, 8, 12] = 1.0
    det_loss = detection_loss(outputs, targets)
    assert torch.isfinite(det_loss["total"])


def test_depth_loss_is_finite(model_outputs: dict[str, torch.Tensor]) -> None:
    outputs = model_outputs
    depth = torch.zeros(1, 16, 24)
    mask = torch.zeros(1, 16, 24, dtype=torch.bool)
    depth[0, 8, 12], mask[0, 8, 12] = 12.0, True
    dep_loss = sparse_depth_loss(outputs["depth_logits"], depth, mask)
    assert torch.isfinite(dep_loss)


def test_decode_does_not_crash(model_outputs: dict[str, torch.Tensor]) -> None:
    outputs = model_outputs
    decoded = decode_detections(
        outputs["heatmap"], outputs["offset"], outputs["size"],
        top_k=5, score_threshold=0.0, image_size=(96, 64),
    )
    assert len(decoded) == 1
    assert decoded[0]["boxes"].shape == (5, 4)
