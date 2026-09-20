"""Chồng LiDAR frame hiện tại lên camera trước để kiểm tra extrinsic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.data.depth_projection import load_pcd_xyz
from src.data.projection import lidar_to_camera, project_camera_points
from src.utils.visualization import overlay_sparse_depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/valid_samples.json"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sample = json.loads(args.index.read_text(encoding="utf-8"))["samples"][args.sample_index]
    image = cv2.imread(sample["camera_path"])
    points = load_pcd_xyz(sample["lidar_path"])
    camera_points = lidar_to_camera(
        points,
        np.asarray(sample["sensor2lidar_rotation"], dtype=np.float32),
        np.asarray(sample["sensor2lidar_translation"], dtype=np.float32),
    )
    pixels, in_front = project_camera_points(camera_points, np.asarray(sample["intrinsic"], dtype=np.float32))
    height, width = image.shape[:2]
    depth = camera_points[:, 2]
    valid = (
        in_front & (depth >= 1.0) & (depth <= 80.0)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    rendered = overlay_sparse_depth(image, pixels[valid], depth[valid])
    output = args.output or Path(f"artifacts/sparse_depth_{args.sample_index:03d}.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), rendered)
    print(f"saved {output} with {int(valid.sum())} projected LiDAR points")
    print(f"positive depth range: {depth[valid].min():.2f}..{depth[valid].max():.2f} m")


if __name__ == "__main__":
    main()
