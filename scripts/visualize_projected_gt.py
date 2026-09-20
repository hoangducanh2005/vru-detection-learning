"""Lưu ảnh camera trước cùng projected GT boxes để kiểm tra extrinsic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401
from src.utils.visualization import CLASS_NAMES, draw_boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("artifacts/valid_samples.json"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sample = json.loads(args.index.read_text(encoding="utf-8"))["samples"][args.sample_index]
    image = cv2.imread(sample["camera_path"])
    boxes = np.asarray([obj["box_xyxy"] for obj in sample["objects"]], dtype=np.float32).reshape(-1, 4)
    labels = np.asarray([obj["class_id"] for obj in sample["objects"]], dtype=np.int64)
    texts = [f"{CLASS_NAMES[obj['class_id']]} ({obj['original_name']})" for obj in sample["objects"]]
    rendered = draw_boxes(image, boxes, labels, texts)
    output = args.output or Path(f"artifacts/projected_gt_{args.sample_index:03d}.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), rendered)
    print(f"saved {output} with {len(boxes)} boxes")


if __name__ == "__main__":
    main()
