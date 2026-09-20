"""Sinh heatmap, center-offset và size target theo phong cách CenterNet."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


# -----------------------------------------------------------------------------
# KHỐI 1: Tính bán kính và vẽ Gaussian quanh tâm object
# -----------------------------------------------------------------------------
def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> int:
    """Tính bán kính Gaussian sao cho overlap với bbox xấp xỉ ``min_overlap``."""

    a1, b1 = 1.0, height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    radius1 = (b1 + math.sqrt(max(0.0, b1**2 - 4 * a1 * c1))) / 2

    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    radius2 = (b2 + math.sqrt(max(0.0, b2**2 - 4 * a2 * c2))) / 2

    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    radius3 = (b3 + math.sqrt(max(0.0, b3**2 - 4 * a3 * c3))) / (2 * a3)
    return max(0, int(min(radius1, radius2, radius3)))


def draw_gaussian(heatmap: npt.NDArray[np.float32], center: tuple[int, int], radius: int) -> None:
    """Vẽ Gaussian tại chỗ; vùng chồng nhau được gộp bằng giá trị lớn nhất."""

    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coordinates = np.arange(diameter, dtype=np.float32) - radius
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)

    x, y = center
    height, width = heatmap.shape
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)
    image_slice = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]
    kernel_slice = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(image_slice, kernel_slice, out=image_slice)


# -----------------------------------------------------------------------------
# KHỐI 2: Chuyển danh sách bbox thành bốn tensor target stride-4
# -----------------------------------------------------------------------------
def build_centernet_targets(
    boxes_xyxy: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    image_size: tuple[int, int],
    output_stride: int = 4,
    num_classes: int = 2,
) -> dict[str, npt.NDArray]:
    """Tạo target dễ học ở duy nhất một feature level stride-4.

    ``heatmap`` trả lời object ở đâu/lớp gì; ``offset`` sửa sai số làm tròn tâm;
    ``size`` lưu width/height; ``mask`` đánh dấu cell được tính L1 regression.
    """

    image_width, image_height = image_size
    output_width, output_height = image_width // output_stride, image_height // output_stride
    heatmap = np.zeros((num_classes, output_height, output_width), dtype=np.float32)
    offset = np.zeros((2, output_height, output_width), dtype=np.float32)
    size = np.zeros((2, output_height, output_width), dtype=np.float32)
    mask = np.zeros((1, output_height, output_width), dtype=np.float32)

    for box, label in zip(np.asarray(boxes_xyxy), np.asarray(labels)):
        x1, y1, x2, y2 = box.astype(np.float32) / output_stride
        box_width, box_height = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center_int_x, center_int_y = int(center_x), int(center_y)
        if not (0 <= center_int_x < output_width and 0 <= center_int_y < output_height):
            continue
        radius = gaussian_radius(box_height, box_width)
        draw_gaussian(heatmap[int(label)], (center_int_x, center_int_y), radius)
        offset[:, center_int_y, center_int_x] = [center_x - center_int_x, center_y - center_int_y]
        size[:, center_int_y, center_int_x] = [box_width, box_height]
        mask[:, center_int_y, center_int_x] = 1.0

    return {"heatmap": heatmap, "offset": offset, "size": size, "mask": mask}
