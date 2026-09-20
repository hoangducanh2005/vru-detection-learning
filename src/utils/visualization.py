"""Các hàm OpenCV nhỏ dùng cho sanity-check và demo."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


CLASS_NAMES = {0: "VEHICLE", 1: "VRU"}
CLASS_COLORS = {0: (40, 180, 40), 1: (40, 40, 230)}  # OpenCV dùng thứ tự BGR.


def draw_boxes(
    image_bgr: npt.NDArray[np.uint8],
    boxes: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer],
    texts: list[str] | None = None,
) -> npt.NDArray[np.uint8]:
    """Vẽ bbox và text; luôn copy để không sửa ảnh đầu vào ngoài ý muốn."""
    output = image_bgr.copy()
    for index, (box, label_value) in enumerate(zip(boxes, labels)):
        label = int(label_value)
        x1, y1, x2, y2 = map(int, np.round(box))
        color = CLASS_COLORS[label]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = texts[index] if texts is not None else CLASS_NAMES[label]
        cv2.putText(output, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def overlay_sparse_depth(
    image_bgr: npt.NDArray[np.uint8],
    pixels_uv: npt.NDArray[np.floating],
    depths: npt.NDArray[np.floating],
    max_depth_m: float = 80.0,
) -> npt.NDArray[np.uint8]:
    """Tô màu các LiDAR point theo depth rồi chồng lên ảnh camera."""
    output = image_bgr.copy()
    normalized = np.clip(np.asarray(depths) / max_depth_m, 0.0, 1.0)
    colors = cv2.applyColorMap((255 * (1.0 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)
    for (u, v), color in zip(pixels_uv, colors[:, 0, :]):
        cv2.circle(output, (int(u), int(v)), 2, tuple(map(int, color)), -1, cv2.LINE_AA)
    return output
