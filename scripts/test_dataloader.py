"""Smoke-test lazy loading, target generation và sparse depth tùy chọn."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from _common import load_config
from src.data.navsim_vru_dataset import NavsimVruDataset, navsim_collate


def main() -> None:
    # KHỐI 1: Dataset + DataLoader Windows-safe với num_workers=0.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/valid_samples.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--with-depth", action="store_true")
    args = parser.parse_args()
    config = load_config()
    data = config["data"]
    model = config["model"]
    dataset = NavsimVruDataset(
        args.index,
        (data["image_width"], data["image_height"]),
        data["output_stride"],
        include_depth=args.with_depth,
        depth_range=(model["depth_min_m"], model["depth_max_m"]),
        max_samples=args.batch_size,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=navsim_collate)))
    # KHỐI 2: In tensor contract và kiểm tra lỗi số học trước khi train.
    targets = batch["targets"]
    labels = torch.cat(batch["raw_labels"])
    tensors = [batch["image"], *targets.values()]
    print("image shape:", tuple(batch["image"].shape))
    print("heatmap shape:", tuple(targets["heatmap"].shape))
    print("number of VEHICLE:", int((labels == 0).sum()))
    print("number of VRU:", int((labels == 1).sum()))
    print("NaN / Inf:", any(not torch.isfinite(t.float()).all() for t in tensors))
    if args.with_depth:
        print("depth shape:", tuple(targets["depth"].shape))
        print("valid LiDAR pixels:", int(targets["depth_mask"].sum()))
        positive = targets["depth"][targets["depth_mask"]]
        print("depth min/max:", float(positive.min()), float(positive.max()))


if __name__ == "__main__":
    main()
