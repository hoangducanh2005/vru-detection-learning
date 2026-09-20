"""Chạy một frame thật qua data, model, decode và distance của Phase 2."""

from pathlib import Path

import pytest
import torch

from src.data.navsim_vru_dataset import NavsimVruDataset
from src.model.depth_head import depth_expectation
from src.model.vru_model import VruModel
from src.utils.decode import decode_detections
from src.utils.distance import estimate_box_distance, vru_risk


def test_end_to_end_real_frame_demo_path() -> None:
    index = Path("artifacts/valid_samples.json")
    if not index.exists():
        pytest.skip("run python scripts/build_index.py first")
    # Resize nhỏ giúp test nhanh; cùng code path này mặc định chạy ở 768x432.
    item = NavsimVruDataset(index, image_size=(192, 108), max_samples=1)[0]
    model = VruModel(fpn_channels=32, with_depth=True, depth_bins=16)
    model.eval()
    with torch.inference_mode():
        outputs = model(item["image"].unsqueeze(0))
        detection = decode_detections(
            outputs["heatmap"], outputs["offset"], outputs["size"],
            top_k=3, score_threshold=0.0, image_size=(192, 108),
        )[0]
        depth = depth_expectation(outputs["depth_logits"], 1.0, 80.0)[0]
    assert len(detection["boxes"]) == 3
    distance = estimate_box_distance(depth, detection["boxes"][0], (192, 108))
    assert 1.0 <= distance <= 80.0
    assert vru_risk(distance) in {"CRITICAL", "NEAR", "FAR"}
