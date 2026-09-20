"""Depth head dạng phân loại bin và phép đổi sang metric depth."""

from __future__ import annotations

import torch
from torch import nn


class DepthHead(nn.Module):
    """Dự đoán một phân phối xác suất trên các khoảng depth tại mỗi pixel."""
    def __init__(self, channels: int = 128, num_bins: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_bins, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network(feature)  # [B,64,H/4,W/4], depth-bin logits


def depth_expectation(logits: torch.Tensor, min_depth_m: float, max_depth_m: float) -> torch.Tensor:
    """Đổi xác suất bin thành camera-Z depth theo mét ``[B,H,W]``.

    Weighted expectation giữ thông tin của toàn phân phối và cho depth liên tục,
    thay vì chỉ chọn cứng một bin bằng argmax.
    """

    bins = logits.shape[1]
    centers = torch.linspace(
        min_depth_m + (max_depth_m - min_depth_m) / (2 * bins),
        max_depth_m - (max_depth_m - min_depth_m) / (2 * bins),
        bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    probabilities = logits.softmax(dim=1)
    return (probabilities * centers.view(1, -1, 1, 1)).sum(dim=1)
