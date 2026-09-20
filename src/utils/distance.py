"""Lấy depth của object và áp dụng near-VRU rule tối giản."""

from __future__ import annotations

import math

import numpy as np
import torch


def estimate_box_distance(
    depth_map: torch.Tensor | np.ndarray,
    box_xyxy: torch.Tensor | np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Lấy median vùng lower-middle của bbox, fallback về center pixel.

    Lower-middle thường chứa thân/chân object và ít background hơn toàn bbox.
    Median bền vững hơn mean trước một số pixel depth ngoại lai.
    """

    depth = np.asarray(depth_map.detach().cpu() if isinstance(depth_map, torch.Tensor) else depth_map)
    box = np.asarray(box_xyxy.detach().cpu() if isinstance(box_xyxy, torch.Tensor) else box_xyxy, dtype=float)
    image_width, image_height = image_size
    map_height, map_width = depth.shape
    x1, y1, x2, y2 = box
    # Lấy 50% giữa theo chiều ngang và 40% dưới theo chiều dọc để tập trung vào
    # vùng thân/chân, đồng thời loại bớt background ở phía trên bbox.
    rx1, rx2 = x1 + 0.25 * (x2 - x1), x2 - 0.25 * (x2 - x1)
    ry1, ry2 = y1 + 0.60 * (y2 - y1), y2
    ix1 = int(np.clip(np.floor(rx1 / image_width * map_width), 0, map_width - 1))
    ix2 = int(np.clip(np.ceil(rx2 / image_width * map_width), ix1 + 1, map_width))
    iy1 = int(np.clip(np.floor(ry1 / image_height * map_height), 0, map_height - 1))
    iy2 = int(np.clip(np.ceil(ry2 / image_height * map_height), iy1 + 1, map_height))
    values = depth[iy1:iy2, ix1:ix2]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size:
        return float(np.median(values))
    center_x = int(np.clip((x1 + x2) / 2 / image_width * map_width, 0, map_width - 1))
    center_y = int(np.clip((y1 + y2) / 2 / image_height * map_height, 0, map_height - 1))
    value = float(depth[center_y, center_x])
    return value if math.isfinite(value) and value > 0 else float("nan")


def vru_risk(distance_m: float, critical_m: float = 10.0, near_m: float = 20.0) -> str:
    """Phân loại khoảng cách demo; đây không phải logic an toàn production."""
    if not math.isfinite(distance_m):
        return "UNKNOWN"
    if distance_m <= critical_m:
        return "CRITICAL"
    if distance_m <= near_m:
        return "NEAR"
    return "FAR"
