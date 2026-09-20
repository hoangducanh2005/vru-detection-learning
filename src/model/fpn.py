"""Feature Pyramid Network top-down tối giản, trả một output stride-4."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SimpleFPN(nn.Module):
    """Đưa C2..C5 về cùng số channel và truyền semantic từ sâu xuống nông."""

    def __init__(self, in_channels: tuple[int, ...] = (64, 128, 256, 512), out_channels: int = 128) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(channels, out_channels, 1) for channels in in_channels])
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # KHỐI 1: Lateral 1x1 conv thống nhất số channel.
        c2, c3, c4, c5 = features
        p5 = self.lateral[3](c5)
        # KHỐI 2: Top-down fusion. ``nearest`` đủ rõ ràng cho bản educational.
        p4 = self.lateral[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lateral[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        # KHỐI 3: Chỉ giữ stride-4 để detector đơn giản. Nhờ các phép cộng
        # top-down, p2 vẫn mang context từ C3/C4/C5 chứ không chỉ feature nông.
        return self.smooth[0](p2)  # [B,out_channels,H/4,W/4]
