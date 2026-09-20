"""Đọc PCD lazy và tạo supervision metric-depth thưa từ LiDAR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .projection import lidar_to_camera, project_camera_points, scale_intrinsic


# -----------------------------------------------------------------------------
# KHỐI 1: Đọc XYZ từ file PCD khi Dataset thực sự cần sample
# -----------------------------------------------------------------------------
def load_pcd_xyz(path: str | Path) -> npt.NDArray[np.float32]:
    """Đọc XYZ từ PCD mà không phụ thuộc toàn bộ thư viện nuPlan.

    NAVSIM mini dùng binary PCD với các field ``x y z intensity lidar_info ring``.
    Parser cũng hỗ trợ ASCII PCD. Hàm chỉ chạy trong ``Dataset.__getitem__`` khi
    cần depth, vì vậy không tạo cache point-cloud hàng chục GB.
    """

    path = Path(path)
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            decoded = line.decode("ascii").strip()
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break

        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        points = int(header["POINTS"][0])
        data_kind = header["DATA"][0].lower()

        if data_kind == "ascii":
            values = np.loadtxt(stream, dtype=np.float32, max_rows=points)
            indices = [fields.index(axis) for axis in ("x", "y", "z")]
            return np.asarray(values[:, indices], dtype=np.float32)
        if data_kind != "binary":
            raise ValueError(f"Unsupported PCD DATA {data_kind!r}; expected binary/ascii")

        code = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2",
                ("U", 4): "<u4", ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}
        dtype_fields = []
        for field, size, value_type, count in zip(fields, sizes, types, counts):
            base = code.get((value_type.upper(), size))
            if base is None:
                raise ValueError(f"Unsupported PCD field type: {value_type}{size}")
            dtype_fields.append((field, base) if count == 1 else (field, base, (count,)))
        records = np.frombuffer(stream.read(), dtype=np.dtype(dtype_fields), count=points)
        return np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float32)


# -----------------------------------------------------------------------------
# KHỐI 2: Project LiDAR và rasterize sparse depth map
# -----------------------------------------------------------------------------
def sparse_depth_map(
    points_lidar: npt.NDArray[np.floating],
    sensor2lidar_rotation: npt.NDArray[np.floating],
    sensor2lidar_translation: npt.NDArray[np.floating],
    intrinsic: npt.NDArray[np.floating],
    source_image_size: tuple[int, int],
    output_size: tuple[int, int],
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Rasterize camera-Z depth, giữ điểm LiDAR gần nhất tại mỗi pixel.

    Nhiều tia LiDAR có thể rơi cùng một pixel sau khi downsample. Chọn depth nhỏ
    nhất tương đương giữ bề mặt nhìn thấy gần camera thay vì bề mặt phía sau.
    """

    camera_points = lidar_to_camera(
        points_lidar, sensor2lidar_rotation, sensor2lidar_translation
    )
    output_intrinsic = scale_intrinsic(intrinsic, source_image_size, output_size)
    pixels, in_front = project_camera_points(camera_points, output_intrinsic)
    depths = camera_points[:, 2]
    output_width, output_height = output_size
    u = np.floor(pixels[:, 0]).astype(np.int64)
    v = np.floor(pixels[:, 1]).astype(np.int64)
    valid = (
        in_front
        & (depths >= min_depth_m)
        & (depths <= max_depth_m)
        & (u >= 0) & (u < output_width)
        & (v >= 0) & (v < output_height)
    )

    depth_map = np.full((output_height, output_width), np.inf, dtype=np.float32)
    flat_indices = v[valid] * output_width + u[valid]
    np.minimum.at(depth_map.reshape(-1), flat_indices, depths[valid])
    mask = np.isfinite(depth_map)
    depth_map[~mask] = 0.0
    return depth_map, mask
