"""
Unit tests for Continual Classifier and ResNet-18 Backbone.
Verifies forward(x) and forward_features(x) interfaces.
"""

import torch
import pytest
from models.classifier import ContinualClassifier


def test_model_forward_and_features():
    # Use randomly initialized backbone for fast offline testing
    model = ContinualClassifier(num_classes=50, pretrained=False)
    dummy_input = torch.randn(2, 3, 128, 128)

    # 1. Test forward_features(x)
    features = model.forward_features(dummy_input)
    assert features.shape == (2, 512), f"Expected shape (2, 512), got {features.shape}"

    # 2. Test forward(x)
    logits = model.forward(dummy_input)
    assert logits.shape == (2, 50), f"Expected shape (2, 50), got {logits.shape}"

    # 3. Test forward_with_features(x)
    l, f = model.forward_with_features(dummy_input)
    assert l.shape == (2, 50)
    assert f.shape == (2, 512)
