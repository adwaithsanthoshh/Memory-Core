"""
Continual Classifier Architecture for MEMORYCORE.
Integrates ResNet18Backbone with configurable classification head.
Strictly implements forward_features(x) and forward(x).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from models.backbone import ResNet18Backbone


class ContinualClassifier(nn.Module):
    """
    Standard model for continual learning experiments.
    
    Architecture:
        Input image (B, 3, 128, 128)
             ↓
        ResNet-18 Backbone
             ↓
        Feature representation (B, 512)
             ↓
        Classification Head (B, num_classes)
    """
    def __init__(
        self,
        num_classes: int = 50,
        pretrained: bool = True,
        freeze_features: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = ResNet18Backbone(
            pretrained=pretrained,
            freeze_features=freeze_features
        )
        self.head = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dimensional feature embedding before classification.
        Exposed per hackathon specification for continual-learning methods.
        """
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass yielding classification logits for all classes.
        """
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both logits and feature embeddings.
        Useful for distillation, replay feature matching, and gradient calculations.
        """
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits, feats

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the architecture and weights."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": "ResNet-18",
            "feature_dim": self.backbone.feature_dim,
            "num_classes": self.num_classes,
            "weights_info": self.backbone.weights_info,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }
