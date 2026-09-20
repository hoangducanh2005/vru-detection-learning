"""Training loop Phase 1 tối giản: raw PyTorch, một feature level."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401
from _common import choose_device, load_config
from src.data.navsim_vru_dataset import NavsimVruDataset, navsim_collate
from src.losses.detection_loss import detection_loss
from src.model.vru_model import VruModel


def main() -> None:
    # KHỐI 1: Đọc tham số và dựng Dataset/DataLoader.
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/valid_samples.json")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/phase1_detector.pt"))
    args = parser.parse_args()
    config = load_config()
    data_cfg, model_cfg, loss_cfg = config["data"], config["model"], config["loss"]
    device = choose_device(args.device)
    dataset = NavsimVruDataset(
        args.index, (data_cfg["image_width"], data_cfg["image_height"]), data_cfg["output_stride"],
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=navsim_collate)
    # KHỐI 2: Model và optimizer thuần PyTorch để workflow không bị framework che.
    model = VruModel(model_cfg["fpn_channels"], pretrained_backbone=model_cfg["pretrained_backbone"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    final_losses: dict[str, float] = {}
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            # KHỐI 3: Forward -> loss -> backward -> update cho một mini-batch.
            image = batch["image"].to(device)
            targets = {key: value.to(device) for key, value in batch["targets"].items()}
            losses = detection_loss(model(image), targets, loss_cfg["lambda_offset"], loss_cfg["lambda_size"])
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            final_losses = {key: float(value.detach()) for key, value in losses.items()}
            print(f"epoch={epoch + 1} step={step + 1} " + " ".join(f"{k}={v:.4f}" for k, v in final_losses.items()))
    # KHỐI 4: Lưu state_dict cùng config/loss để checkpoint tự mô tả.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "final_losses": final_losses}, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
