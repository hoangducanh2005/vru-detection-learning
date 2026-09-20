"""Dataset nhỏ, lazy-load, dành cho camera trước của NAVSIM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .depth_projection import load_pcd_xyz, sparse_depth_map
from .targets import build_centernet_targets


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class NavsimVruDataset(Dataset[dict[str, Any]]):
    """Đọc pixel/point-cloud khi cần từ một JSON index gọn nhẹ.

    Index chỉ chứa path, calibration và projected labels. Thiết kế này tránh lỗi
    phổ biến là decode toàn bộ sensor rồi cache thành file nhiều chục GB.
    """

    def __init__(
        self,
        index_path: str | Path = "artifacts/valid_samples.json",
        image_size: tuple[int, int] = (768, 432),
        output_stride: int = 4,
        include_depth: bool = False,
        depth_range: tuple[float, float] = (1.0, 80.0),
        max_samples: int | None = None,
    ) -> None:
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.samples = payload["samples"][:max_samples]
        self.image_size = image_size  # (width, height)
        self.output_stride = output_stride
        self.include_depth = include_depth
        self.depth_range = depth_range

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # ---------------------------------------------------------------------
        # KHỐI 1: Lazy-load và resize duy nhất camera trước
        # ---------------------------------------------------------------------
        sample = self.samples[index]
        bgr = cv2.imread(sample["camera_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(sample["camera_path"])
        source_height, source_width = bgr.shape[:2]
        target_width, target_height = self.image_size
        interpolation = cv2.INTER_AREA if target_width < source_width else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(
            cv2.resize(bgr, self.image_size, interpolation=interpolation), cv2.COLOR_BGR2RGB
        )

        # ---------------------------------------------------------------------
        # KHỐI 2: Scale bbox và sinh CenterNet targets
        # ---------------------------------------------------------------------
        sx, sy = target_width / source_width, target_height / source_height
        boxes = np.asarray([obj["box_xyxy"] for obj in sample["objects"]], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray([obj["class_id"] for obj in sample["objects"]], dtype=np.int64)
        if len(boxes):
            boxes *= np.array([sx, sy, sx, sy], dtype=np.float32)
        targets_np = build_centernet_targets(
            boxes, labels, self.image_size, self.output_stride, num_classes=2
        )

        # ---------------------------------------------------------------------
        # KHỐI 3: Chuẩn hóa ảnh và đóng gói sample PyTorch
        # ---------------------------------------------------------------------
        normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()  # [3,H,W]
        result: dict[str, Any] = {
            "image": image_tensor,
            "targets": {key: torch.from_numpy(value) for key, value in targets_np.items()},
            "raw_boxes": torch.from_numpy(boxes),
            "raw_labels": torch.from_numpy(labels),
            "meta": {"token": sample["token"], "camera_path": sample["camera_path"]},
        }

        if self.include_depth:
            # ---------------------------------------------------------------
            # KHỐI 4: Chỉ đọc PCD khi Phase 2 yêu cầu depth supervision
            # ---------------------------------------------------------------
            points = load_pcd_xyz(sample["lidar_path"])
            feature_size = (target_width // self.output_stride, target_height // self.output_stride)
            depth, depth_mask = sparse_depth_map(
                points,
                np.asarray(sample["sensor2lidar_rotation"], dtype=np.float32),
                np.asarray(sample["sensor2lidar_translation"], dtype=np.float32),
                np.asarray(sample["intrinsic"], dtype=np.float32),
                tuple(sample["image_size"]),
                feature_size,
                *self.depth_range,
            )
            result["targets"]["depth"] = torch.from_numpy(depth)
            result["targets"]["depth_mask"] = torch.from_numpy(depth_mask)
        return result


def navsim_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor cố định, giữ raw bbox độ dài biến đổi dưới dạng list.

    PyTorch không thể stack ``[N,4]`` khi mỗi ảnh có N khác nhau. Detection
    targets vẫn có kích thước cố định nên được stack bình thường.
    """

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "targets": {
            key: torch.stack([item["targets"][key] for item in batch])
            for key in batch[0]["targets"]
        },
        "raw_boxes": [item["raw_boxes"] for item in batch],
        "raw_labels": [item["raw_labels"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
