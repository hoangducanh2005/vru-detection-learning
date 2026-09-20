"""Tạo JSON index nhỏ chứa metadata/label cho PyTorch Dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import cv2
import numpy as np
import yaml

import _bootstrap  # noqa: F401
from src.data.projection import project_boxes_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path(r"D:\navsim_workspace\dataset\navsim_logs\mini"))
    parser.add_argument("--sensors", type=Path, default=Path(r"D:\navsim_workspace\dataset\sensor_blobs\mini"))
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/taxonomy.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/valid_samples.json"))
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def load_taxonomy(path: Path) -> dict[str, int]:
    """Đổi taxonomy dạng tên -> nhóm thành tên -> class id số nguyên."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    class_ids = config["classes"]
    return {
        name: int(class_ids[target])
        for name, target in config["mapping"].items()
        if target != "IGNORE"
    }


def main() -> None:
    # KHỐI 1: Đọc taxonomy do người dùng kiểm soát, không đoán class bằng size.
    args = parse_args()
    taxonomy = load_taxonomy(args.taxonomy)
    samples: list[dict] = []
    for log_path in sorted(args.logs.glob("*.pkl")):
        # KHỐI 2: Chỉ đọc pickle gốc của NAVSIM; tuyệt đối không sinh pickle cache.
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            camera_meta = frame["cams"]["CAM_F0"]
            camera_path = (args.sensors / camera_meta["data_path"]).resolve()
            lidar_path = (args.sensors / frame["lidar_path"]).resolve()
            if not camera_path.is_file() or not lidar_path.is_file():
                continue

            # Đọc từng ảnh để biết kích thước rồi giải phóng ở vòng lặp tiếp theo;
            # không giữ decoded pixels của toàn dataset trong RAM hoặc index.
            image = cv2.imread(str(camera_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            annotations = frame["anns"]
            projected = project_boxes_to_image(
                annotations["gt_boxes"], annotations["gt_names"], taxonomy,
                camera_meta["sensor2lidar_rotation"], camera_meta["sensor2lidar_translation"],
                camera_meta["cam_intrinsic"], (width, height),
            )
            samples.append(
                {
                    "token": str(frame["token"]),
                    "log_name": str(frame["log_name"]),
                    "camera_path": str(camera_path),
                    "lidar_path": str(lidar_path),
                    "image_size": [width, height],
                    "intrinsic": np.asarray(camera_meta["cam_intrinsic"]).tolist(),
                    "sensor2lidar_rotation": np.asarray(camera_meta["sensor2lidar_rotation"]).tolist(),
                    "sensor2lidar_translation": np.asarray(camera_meta["sensor2lidar_translation"]).tolist(),
                    "objects": [
                        {
                            "box_xyxy": list(item.box_xyxy),
                            "class_id": item.class_id,
                            "original_name": item.original_name,
                        }
                        for item in projected
                    ],
                }
            )
            if args.max_samples is not None and len(samples) >= args.max_samples:
                break
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break

    # KHỐI 3: JSON chỉ lưu path, calibration và label nhỏ gọn.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "description": "Paths, calibration and projected 2D labels only; pixels/point clouds are lazy-loaded.",
        "source_logs": str(args.logs.resolve()),
        "source_sensors": str(args.sensors.resolve()),
        "samples": samples,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    vehicle_count = sum(o["class_id"] == 0 for s in samples for o in s["objects"])
    vru_count = sum(o["class_id"] == 1 for s in samples for o in s["objects"])
    print(f"wrote {len(samples)} valid samples to {args.output}")
    print(f"projected boxes: VEHICLE={vehicle_count}, VRU={vru_count}")


if __name__ == "__main__":
    main()
