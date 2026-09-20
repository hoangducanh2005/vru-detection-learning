"""Train Phase 2 với shared FPN, detection loss và sparse depth loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from _common import choose_device, load_config, load_model_state
from src.data.navsim_vru_dataset import NavsimVruDataset, navsim_collate
from src.losses.depth_loss import sparse_depth_loss
from src.losses.detection_loss import detection_loss
from src.model.vru_model import VruModel


def main() -> None:
    # KHỐI 1: Phase 2 bật lazy LiDAR loading và depth targets.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/valid_samples.json")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--load-phase1", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/phase2_multitask.pt"))
    args = parser.parse_args()
    config = load_config()
    data_cfg, model_cfg, loss_cfg = config["data"], config["model"], config["loss"]
    depth_range = (model_cfg["depth_min_m"], model_cfg["depth_max_m"])
    device = choose_device(args.device)
    dataset = NavsimVruDataset(
        args.index, (data_cfg["image_width"], data_cfg["image_height"]), data_cfg["output_stride"],
        include_depth=True, depth_range=depth_range, max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=navsim_collate)
    model = VruModel(
        model_cfg["fpn_channels"], with_depth=True, depth_bins=model_cfg["depth_bins"],
        pretrained_backbone=model_cfg["pretrained_backbone"],
    )
    if args.load_phase1:
        # Chỉ depth head bị thiếu khi nạp Phase 1; backbone/FPN/detector được giữ.
        load_model_state(model, args.load_phase1, strict=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    final_losses: dict[str, float] = {}
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            # KHỐI 2: Hai task dùng chung một forward và được cộng theo lambda_depth.
            image = batch["image"].to(device)
            targets = {key: value.to(device) for key, value in batch["targets"].items()}
            outputs = model(image)
            det = detection_loss(outputs, targets, loss_cfg["lambda_offset"], loss_cfg["lambda_size"])
            depth = sparse_depth_loss(outputs["depth_logits"], targets["depth"], targets["depth_mask"], *depth_range)
            total = det["total"] + loss_cfg["lambda_depth"] * depth
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            final_losses = {"total": float(total.detach()), "detection": float(det["total"].detach()), "depth": float(depth.detach())}
            print(f"epoch={epoch + 1} step={step + 1} " + " ".join(f"{k}={v:.4f}" for k, v in final_losses.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "final_losses": final_losses}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
