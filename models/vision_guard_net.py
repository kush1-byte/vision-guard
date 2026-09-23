"""
The model.

A shared pre-trained backbone (EfficientNet-B0 or ResNet-50) extracts features,
then TWO small heads sit on top:

  * disease_head   -> 5 outputs (one per disease). These are logits; at inference
                      we apply a sigmoid to each independently, so an image can be
                      positive for several diseases at once (multi-LABEL).
  * severity_head  -> 5 outputs for DR severity stage. These compete with each
                      other (softmax), because an image has exactly one DR stage
                      (multi-CLASS).

We use `timm` to load the backbone because it makes swapping backbones a
one-word change and gives us the feature vector directly (num_classes=0).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class VisionGuardNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", num_diseases=5,
                 num_severity=5, pretrained=True, dropout=0.3):
        super().__init__()

        # num_classes=0 -> the backbone returns a pooled feature vector, not
        # class scores. global_pool="avg" gives us one vector per image.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        self.dropout = nn.Dropout(dropout)
        self.disease_head = nn.Linear(feat_dim, num_diseases)
        self.severity_head = nn.Linear(feat_dim, num_severity)

    def forward(self, x):
        features = self.backbone(x)          # (B, feat_dim)
        features = self.dropout(features)
        return {
            "disease_logits": self.disease_head(features),    # (B, 5)
            "severity_logits": self.severity_head(features),  # (B, 5)
        }


def get_target_layers(model: VisionGuardNet, backbone: str):
    """The convolutional layer Grad-CAM should hook into (the last one before
    pooling gives the sharpest heatmaps). Different backbones name it differently.
    """
    if backbone.startswith("efficientnet"):
        return [model.backbone.conv_head]
    if backbone.startswith("resnet"):
        return [model.backbone.layer4[-1]]
    # Fallback: last module that has parameters. Adjust if you use another net.
    return [list(model.backbone.modules())[-1]]
