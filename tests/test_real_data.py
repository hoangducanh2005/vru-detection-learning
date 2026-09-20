"""Integration test chạy trực tiếp trên NAVSIM mini local."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.depth_projection import load_pcd_xyz, sparse_depth_map
from src.data.navsim_vru_dataset import NavsimVruDataset


INDEX = Path("artifacts/valid_samples.json")


def _sample() -> dict:
    if not INDEX.exists():
        pytest.skip("run python scripts/build_index.py first")
    return json.loads(INDEX.read_text(encoding="utf-8"))["samples"][0]


def test_load_one_real_navsim_sample() -> None:
    sample = _sample()
    assert Path(sample["camera_path"]).is_file()
    assert Path(sample["lidar_path"]).is_file()
    assert sample["token"]


def test_real_projected_boxes_are_valid() -> None:
    sample = _sample()
    width, height = sample["image_size"]
    assert sample["objects"]
    for obj in sample["objects"]:
        x1, y1, x2, y2 = obj["box_xyxy"]
        assert 0 <= x1 < x2 < width
        assert 0 <= y1 < y2 < height
        assert obj["class_id"] in (0, 1)


def test_real_lidar_projection_has_positive_depth() -> None:
    sample = _sample()
    points = load_pcd_xyz(sample["lidar_path"])
    depth, mask = sparse_depth_map(
        points,
        np.asarray(sample["sensor2lidar_rotation"]),
        np.asarray(sample["sensor2lidar_translation"]),
        np.asarray(sample["intrinsic"]),
        tuple(sample["image_size"]),
        (192, 108),
    )
    assert mask.any()
    assert np.all(depth[mask] > 0)


def test_dataset_default_shapes() -> None:
    _sample()
    item = NavsimVruDataset(INDEX, max_samples=1)[0]
    assert item["image"].shape == (3, 432, 768)
    assert item["targets"]["heatmap"].shape == (2, 108, 192)
    assert item["targets"]["offset"].shape == (2, 108, 192)
    assert item["targets"]["size"].shape == (2, 108, 192)
    assert item["targets"]["mask"].shape == (1, 108, 192)
