"""
Backbone: ResNet-18 for Continual Learning.
Supports official torchvision ImageNet-pretrained weights (ResNet18_Weights.DEFAULT / IMAGENET1K_V1).
Exposes forward_features(x) to yield representation vectors.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import logging

logger = logging.getLogger("memorycore.models")


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 feature extractor.
    Exposes 512-dimensional feature embedding via forward_features(x).
    """
    WEIGHTS_VERSION = "IMAGENET1K_V1"
    SOURCE = "torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)"

    def __init__(self, pretrained: bool = True, freeze_features: bool = False):
        super().__init__()
        self.pretrained = pretrained
        self.weights_info = None

        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
                base_model = models.resnet18(weights=weights)
                self.weights_info = {
                    "source": self.SOURCE,
                    "version": self.WEIGHTS_VERSION,
                    "url": weights.url,
                    "meta": weights.meta
                }
                logger.info(f"Loaded pretrained ResNet-18 weights ({self.WEIGHTS_VERSION})")
            except Exception as e:
                logger.warning(f"Could not load online pretrained weights ({e}). Initializing randomly.")
                base_model = models.resnet18(weights=None)
                self.weights_info = {"source": "random_initialization", "version": "none"}
        else:
            base_model = models.resnet18(weights=None)
            self.weights_info = {"source": "from_scratch", "version": "none"}

        # Decompose ResNet-18
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.feature_dim = 512

        if freeze_features:
            for p in self.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 512-dimensional feature embedding before classification layer.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
