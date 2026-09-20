"""Demo Phase 2 end-to-end: detection + depth + distance + VRU risk."""

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
from src.model.depth_head import depth_expectation
from src.model.vru_model import VruModel
from src.utils.decode import decode_detections
from src.utils.distance import estimate_box_distance, vru_risk
from src.utils.visualization import CLASS_NAMES, draw_boxes


def main() -> None:
    # KHỐI 1: Nạp cùng một front-camera frame và checkpoint multi-task.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/valid_samples.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/phase2_multitask.pt"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="File JSON; mac dinh artifacts/demo_<index>.json",
    )
    args = parser.parse_args()
    config = load_config()
    data_cfg, model_cfg, infer_cfg = config["data"], config["model"], config["inference"]
    image_size = (data_cfg["image_width"], data_cfg["image_height"])
    dataset = NavsimVruDataset(args.index, image_size, data_cfg["output_stride"])
    sample = dataset[args.sample_index]
    device = choose_device(args.device)
    model = VruModel(model_cfg["fpn_channels"], with_depth=True, depth_bins=model_cfg["depth_bins"]).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()
    # KHỐI 2: Một shared forward sinh cả detection maps và depth-bin logits.
    with torch.inference_mode():
        outputs = model(sample["image"].unsqueeze(0).to(device))
        detection = decode_detections(
            outputs["heatmap"], outputs["offset"], outputs["size"], data_cfg["output_stride"],
            args.top_k or infer_cfg["top_k"],
            args.score_threshold if args.score_threshold is not None else infer_cfg["score_threshold"],
            image_size,
        )[0]
        metric_depth = depth_expectation(
            outputs["depth_logits"], model_cfg["depth_min_m"], model_cfg["depth_max_m"]
        )[0]

    # KHỐI 3: Ghép từng bbox với depth lower-middle và chỉ gán risk cho VRU.
    records: list[dict] = []
    texts: list[str] = []
    for box, score, label_tensor in zip(detection["boxes"], detection["scores"], detection["labels"]):
        label = int(label_tensor)
        distance = estimate_box_distance(metric_depth, box, image_size)
        record = {
            "class": CLASS_NAMES[label],
            "bbox": [round(float(value), 2) for value in box],
            "confidence": round(float(score), 4),
            "distance_m": round(distance, 2),
        }
        if label == 1:
            record["status"] = vru_risk(distance, infer_cfg["critical_distance_m"], infer_cfg["near_distance_m"])
        records.append(record)
        suffix = f" {record['status']}" if "status" in record else ""
        texts.append(f"{record['class']} {float(score):.2f} {distance:.1f}m{suffix}")

    # KHỐI 4: Xuất cả JSON machine-readable và ảnh trực quan.
    boxes_np = detection["boxes"].cpu().numpy()
    labels_np = detection["labels"].cpu().numpy()
    image = resized_bgr(sample["meta"]["camera_path"], image_size)
    rendered = draw_boxes(image, boxes_np, labels_np, texts)
    output = args.output or Path(f"artifacts/demo_{args.sample_index:03d}.jpg")
    json_output = args.json_output or Path(f"artifacts/demo_{args.sample_index:03d}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), rendered)
    result_payload = {
        "sample_index": args.sample_index,
        "token": sample["meta"]["token"],
        "objects": records,
    }
    json_output.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(json.dumps(result_payload, indent=2))
    print(f"saved {output} and {json_output}")


if __name__ == "__main__":
    main()
