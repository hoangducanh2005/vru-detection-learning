"""Masked cross entropy cho categorical metric depth thưa."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_depth_loss(
    logits: torch.Tensor,
    depth_m: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> torch.Tensor:
    """Chỉ tính loss ở pixel thực sự có LiDAR return được project.

    Pixel không có LiDAR không đồng nghĩa depth bằng 0; đó là vùng không có
    supervision và bắt buộc phải bị mask khỏi loss.
    """

    bins = logits.shape[1]
    bin_target = ((depth_m - min_depth_m) / (max_depth_m - min_depth_m) * bins).long()
    bin_target = bin_target.clamp(0, bins - 1)
    per_pixel = F.cross_entropy(logits, bin_target, reduction="none")
    mask = valid_mask.bool() & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    if not mask.any():
        return logits.sum() * 0.0
    return per_pixel[mask].mean()
