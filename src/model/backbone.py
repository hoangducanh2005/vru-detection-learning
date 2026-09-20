"""Tách ResNet34 thành bốn tầng feature C2/C3/C4/C5."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class ResNet34Backbone(nn.Module):
    """Trả feature trung gian thay vì logits phân loại ImageNet.

    Tầng càng sâu có semantic mạnh hơn nhưng độ phân giải thấp hơn. FPN ở bước
    sau sẽ kết hợp hai ưu điểm này cho bài toán phát hiện object nhỏ.
    """

    out_channels = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        network = resnet34(weights=weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # KHỐI FORWARD: ảnh -> bốn mức feature với stride tăng dần.
        # image: [B,3,H,W]
        x = self.stem(image)
        c2 = self.layer1(x)  # [B,64,H/4,W/4]
        c3 = self.layer2(c2)  # [B,128,H/8,W/8]
        c4 = self.layer3(c3)  # [B,256,H/16,W/16]
        c5 = self.layer4(c4)  # [B,512,H/32,W/32]
        return c2, c3, c4, c5
