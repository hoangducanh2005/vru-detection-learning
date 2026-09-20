"""Decode các map CenterNet thành detection trong tọa độ pixel."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_detections(
    heatmap_logits: torch.Tensor,
    offset: torch.Tensor,
    size: torch.Tensor,
    stride: int = 4,
    top_k: int = 100,
    score_threshold: float = 0.25,
    image_size: tuple[int, int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Thực hiện sigmoid -> local max -> top-K -> center/size -> bbox XYXY."""

    # KHỐI 1: Chỉ giữ peak cục bộ 3x3, tương tự một NMS rất nhẹ.
    heatmap = heatmap_logits.sigmoid()
    local_max = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * heatmap.eq(local_max)
    # KHỐI 2: Top-K trên cả class và không gian, rồi tách lại class/x/y.
    batch, classes, height, width = heatmap.shape
    k = min(top_k, classes * height * width)
    scores, indices = torch.topk(heatmap.reshape(batch, -1), k=k)
    labels = indices // (height * width)
    spatial = indices % (height * width)
    ys, xs = spatial // width, spatial % width

    # KHỐI 3: Lấy offset/size đúng tại từng peak và đổi về pixel ảnh gốc.
    results: list[dict[str, torch.Tensor]] = []
    for batch_index in range(batch):
        keep = scores[batch_index] >= score_threshold
        sample_scores = scores[batch_index][keep]
        sample_labels = labels[batch_index][keep]
        sample_spatial = spatial[batch_index][keep]
        sample_x = xs[batch_index][keep].float()
        sample_y = ys[batch_index][keep].float()
        sample_offset = offset[batch_index].reshape(2, -1)[:, sample_spatial].T
        sample_size = size[batch_index].reshape(2, -1)[:, sample_spatial].T.relu()
        centers = torch.stack([sample_x, sample_y], dim=1) + sample_offset
        boxes = torch.cat([centers - sample_size / 2.0, centers + sample_size / 2.0], dim=1) * stride
        if image_size is not None and len(boxes):
            image_width, image_height = image_size
            boxes[:, 0::2].clamp_(0, image_width - 1)
            boxes[:, 1::2].clamp_(0, image_height - 1)
        results.append({"boxes": boxes, "scores": sample_scores, "labels": sample_labels})
    return results
