"""Nạp checkpoint Phase 1, decode một frame NAVSIM và vẽ detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import _bootstrap  # noqa: F401
from _common import choose_device, load_config, load_model_state, resized_bgr
from src.data.navsim_vru_dataset import NavsimVruDataset
from src.model.vru_model import VruModel
from src.utils.decode import decode_detections
from src.utils.visualization import CLASS_NAMES, draw_boxes


def main() -> None:
    # KHỐI 1: Lazy-load một frame và khôi phục đúng model Phase 1.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/valid_samples.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase1_detector.pt"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="File JSON; mac dinh artifacts/phase1_detection_<index>.json",
    )
    args = parser.parse_args()
    config = load_config()
    data_cfg, model_cfg, infer_cfg = config["data"], config["model"], config["inference"]
    image_size = (data_cfg["image_width"], data_cfg["image_height"])
    dataset = NavsimVruDataset(args.index, image_size, data_cfg["output_stride"])
    sample = dataset[args.sample_index]
    device = choose_device(args.device)
    model = VruModel(model_cfg["fpn_channels"]).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()
    # KHỐI 2: inference_mode tắt gradient để tiết kiệm RAM và thời gian.
    with torch.inference_mode():
        outputs = model(sample["image"].unsqueeze(0).to(device))
        detections = decode_detections(
            outputs["heatmap"], outputs["offset"], outputs["size"],
            data_cfg["output_stride"], args.top_k or infer_cfg["top_k"],
            args.score_threshold if args.score_threshold is not None else infer_cfg["score_threshold"],
            image_size,
        )[0]
    # KHỐI 3: Đổi tensor thành kiểu Python thuần để lưu JSON machine-readable.
    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    texts = [f"{CLASS_NAMES[int(label)]} {score:.2f}" for label, score in zip(labels, scores)]
    records = [
        {
            "class": CLASS_NAMES[int(label)],
            "bbox": [round(float(value), 2) for value in box],
            "confidence": round(float(score), 4),
        }
        for box, score, label in zip(boxes, scores, labels)
    ]
    result_payload = {
        "sample_index": args.sample_index,
        "token": sample["meta"]["token"],
        "objects": records,
    }

    # KHỐI 4: Lưu cả ảnh trực quan và file JSON để chương trình khác đọc.
    image = resized_bgr(sample["meta"]["camera_path"], image_size)
    rendered = draw_boxes(image, boxes, labels, texts)
    output = args.output or Path(f"artifacts/phase1_detection_{args.sample_index:03d}.jpg")
    json_output = args.json_output or Path(f"artifacts/phase1_detection_{args.sample_index:03d}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), rendered)
    json_output.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"saved {output} and {json_output}; detections={len(boxes)}")


if __name__ == "__main__":
    main()
