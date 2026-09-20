"""Model Phase 1/2 hoàn chỉnh với image feature dùng chung."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import ResNet34Backbone
from .depth_head import DepthHead
from .detection_head import DetectionHead
from .fpn import SimpleFPN


class VruModel(nn.Module):
    """Ghép backbone, FPN và các task head thành một pipeline dễ theo dõi."""
    def __init__(
        self,
        fpn_channels: int = 128,
        num_classes: int = 2,
        with_depth: bool = False,
        depth_bins: int = 64,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ResNet34Backbone(pretrained_backbone)
        self.fpn = SimpleFPN(self.backbone.out_channels, fpn_channels)
        self.detection_head = DetectionHead(fpn_channels, num_classes)
        self.depth_head = DepthHead(fpn_channels, depth_bins) if with_depth else None

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        # KHỐI 1: Trích xuất và hợp nhất feature đa mức.
        # image: [B,3,H,W]
        feature = self.fpn(self.backbone(image))  # [B,128,H/4,W/4]
        # KHỐI 2: Detection luôn tồn tại ở cả Phase 1 và Phase 2.
        outputs = self.detection_head(feature)
        outputs["feature"] = feature
        # KHỐI 3: Depth là nhánh tùy chọn và dùng đúng shared FPN feature.
        if self.depth_head is not None:
            outputs["depth_logits"] = self.depth_head(feature)  # [B,64,H/4,W/4]
        return outputs
