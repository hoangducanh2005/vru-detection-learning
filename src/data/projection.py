"""Chiếu hộp 3D trong hệ LiDAR của NAVSIM lên ảnh camera trước.

Quy ước tọa độ đã được đối chiếu với ``navsim/visualization/camera.py``:

* annotation có dạng ``[x, y, z, length, width, height, heading]``;
* ``sensor2lidar`` biến đổi điểm từ camera sang LiDAR;
* vì dữ liệu đầu vào ở LiDAR, ta dùng ``p_cam = inv(R) @ (p_lidar - t)``;
* điểm nằm trước camera có Z dương, sau đó ``K @ p_cam`` cho tọa độ ảnh.

Code viết phép biến đổi bằng vector cột một cách tường minh. Cách này dài hơn
một chút nhưng giúp người học nhìn rõ chiều transform, tránh lỗi đảo extrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


# -----------------------------------------------------------------------------
# KHỐI 1: Kiểu dữ liệu đầu ra
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectedBox:
    """Một bbox 2D đã clip, sinh ra từ một annotation 3D của NAVSIM."""

    box_xyxy: tuple[float, float, float, float]
    class_id: int
    original_name: str


# -----------------------------------------------------------------------------
# KHỐI 2: Biến đổi hệ tọa độ LiDAR -> camera
# -----------------------------------------------------------------------------
def lidar_to_camera(
    points_lidar: FloatArray,
    sensor2lidar_rotation: FloatArray,
    sensor2lidar_translation: FloatArray,
) -> npt.NDArray[np.float32]:
    """Đổi các điểm ``[N, 3]`` từ hệ LiDAR sang hệ camera.

    NAVSIM lưu chiều ngược lại (camera -> LiDAR), vì vậy bắt buộc phải nghịch
    đảo. Trong hệ camera, Z dương nghĩa là điểm nằm phía trước ống kính.
    """

    points = np.asarray(points_lidar, dtype=np.float32)
    rotation = np.asarray(sensor2lidar_rotation, dtype=np.float32)
    translation = np.asarray(sensor2lidar_translation, dtype=np.float32)
    return (np.linalg.inv(rotation) @ (points - translation).T).T.astype(np.float32)


def project_camera_points(
    points_camera: FloatArray,
    intrinsic: FloatArray,
    min_depth: float = 1e-3,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Chiếu camera XYZ ra pixel, trả về ``uv`` và mask điểm nằm phía trước."""

    points = np.asarray(points_camera, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    in_front = points[:, 2] > min_depth
    uvw = (intrinsic @ points.T).T
    safe_z = np.maximum(uvw[:, 2:3], min_depth)
    uv = uvw[:, :2] / safe_z
    return uv.astype(np.float32), in_front


# -----------------------------------------------------------------------------
# KHỐI 3: Tạo 8 đỉnh hộp 3D
# -----------------------------------------------------------------------------
def box3d_corners_lidar(box: Iterable[float]) -> npt.NDArray[np.float32]:
    """Tạo 8 đỉnh cho hộp ``[x,y,z,length,width,height,heading]``.

    ``heading`` quay quanh trục +Z của LiDAR. Thứ tự kích thước đã được xác nhận
    từ ``BoundingBoxIndex`` và code visualization chính thức của NAVSIM.
    """

    x, y, z, length, width, height, heading = map(float, box)
    signs = np.array(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float32,
    )
    local = signs * np.array([length, width, height], dtype=np.float32) / 2.0
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation_z = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (rotation_z @ local.T).T + np.array([x, y, z], dtype=np.float32)


# -----------------------------------------------------------------------------
# KHỐI 4: Chiếu hộp 3D thành bbox 2D
# -----------------------------------------------------------------------------
def project_boxes_to_image(
    boxes_3d: FloatArray,
    names: Iterable[str],
    taxonomy: dict[str, int],
    sensor2lidar_rotation: FloatArray,
    sensor2lidar_translation: FloatArray,
    intrinsic: FloatArray,
    image_size: tuple[int, int],
    min_area_px: float = 4.0,
) -> list[ProjectedBox]:
    """Chiếu các hộp được hỗ trợ, clip theo ảnh và bỏ hộp không nhìn thấy.

    ``image_size`` có thứ tự ``(width, height)``. Hộp chỉ được giữ khi có ít
    nhất một đỉnh ở trước camera và diện tích sau clip đủ lớn. Với object cắt
    near-plane, ta chỉ dùng các đỉnh phía trước để code dễ đọc và ổn định.
    """

    width, height = image_size
    results: list[ProjectedBox] = []
    for box, name_value in zip(np.asarray(boxes_3d), names):
        name = str(name_value)
        if name not in taxonomy:
            continue
        corners_camera = lidar_to_camera(
            box3d_corners_lidar(box), sensor2lidar_rotation, sensor2lidar_translation
        )
        pixels, in_front = project_camera_points(corners_camera, intrinsic)
        if not np.any(in_front):
            continue
        visible_pixels = pixels[in_front]
        x_min, y_min = visible_pixels.min(axis=0)
        x_max, y_max = visible_pixels.max(axis=0)
        x_min = float(np.clip(x_min, 0.0, width - 1.0))
        y_min = float(np.clip(y_min, 0.0, height - 1.0))
        x_max = float(np.clip(x_max, 0.0, width - 1.0))
        y_max = float(np.clip(y_max, 0.0, height - 1.0))
        if (x_max - x_min) * (y_max - y_min) < min_area_px:
            continue
        results.append(
            ProjectedBox((x_min, y_min, x_max, y_max), taxonomy[name], name)
        )
    return results


# -----------------------------------------------------------------------------
# KHỐI 5: Cập nhật intrinsic khi resize ảnh
# -----------------------------------------------------------------------------
def scale_intrinsic(
    intrinsic: FloatArray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> npt.NDArray[np.float32]:
    """Scale ``fx, fy, cx, cy`` sau khi resize ảnh."""

    source_width, source_height = source_size
    target_width, target_height = target_size
    scaled = np.asarray(intrinsic, dtype=np.float32).copy()
    scaled[0, :] *= target_width / source_width
    scaled[1, :] *= target_height / source_height
    return scaled
