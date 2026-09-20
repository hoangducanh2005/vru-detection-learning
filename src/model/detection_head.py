"""Ba CenterNet head nhỏ cùng đọc shared feature stride-4."""

from __future__ import annotations

import math

import torch
from torch import nn


def _head(in_channels: int, out_channels: int) -> nn.Sequential:
    """Một block Conv-ReLU-Conv dễ đọc, dùng chung cấu trúc cho ba head."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1),
    )


class DetectionHead(nn.Module):
    def __init__(self, channels: int = 128, num_classes: int = 2) -> None:
        super().__init__()
        self.heatmap = _head(channels, num_classes)
        self.offset = _head(channels, 2)
        self.size = _head(channels, 2)
        # Prior 1% giúp heatmap ban đầu không coi mọi cell là object. Nếu bias
        # bằng 0 thì sigmoid=0.5, focal loss ban đầu sẽ rất lớn và khó quan sát.
        nn.init.constant_(self.heatmap[-1].bias, -math.log((1.0 - 0.01) / 0.01))

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        # Ba nhánh độc lập giúp người học nhìn rõ nhiệm vụ của từng tensor.
        return {
            "heatmap": self.heatmap(feature),  # [B,2,H/4,W/4], logits
            "offset": self.offset(feature),  # [B,2,H/4,W/4]
            "size": self.size(feature),  # [B,2,H/4,W/4]
        }
