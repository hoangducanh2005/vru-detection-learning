"""Các loss cho detection và sparse depth."""

from .depth_loss import sparse_depth_loss
from .detection_loss import detection_loss

__all__ = ["detection_loss", "sparse_depth_loss"]
