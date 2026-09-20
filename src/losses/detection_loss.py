"""CenterNet focal loss và hai masked L1 regression loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal loss kiểu CenterNet; vành Gaussian là negative có trọng số nhỏ.

    Object center chính xác có target=1. Các pixel lân cận vẫn là negative nhưng
    được giảm trọng số để model không bị phạt nặng khi dự đoán gần tâm thật.
    """

    prediction = logits.sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    negative_weight = (1.0 - target).pow(4)
    positive_loss = torch.log(prediction) * (1.0 - prediction).pow(2) * positive
    negative_loss = torch.log(1.0 - prediction) * prediction.pow(2) * negative_weight * negative
    positive_count = positive.float().sum().clamp(min=1.0)
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Chỉ hồi quy offset/size tại cell chứa tâm object."""
    expanded_mask = mask.expand_as(prediction)
    return F.l1_loss(prediction * expanded_mask, target * expanded_mask, reduction="sum") / expanded_mask.sum().clamp(min=1.0)


def detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambda_offset: float = 1.0,
    lambda_size: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Ghép ba thành phần loss và trả riêng từng phần để log dễ hiểu."""
    heatmap = heatmap_focal_loss(outputs["heatmap"], targets["heatmap"])
    offset = masked_l1(outputs["offset"], targets["offset"], targets["mask"])
    size = masked_l1(outputs["size"], targets["size"], targets["mask"])
    total = heatmap + lambda_offset * offset + lambda_size * size
    return {"total": total, "heatmap": heatmap, "offset": offset, "size": size}
