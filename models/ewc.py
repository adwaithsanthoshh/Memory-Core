"""
Elastic Weight Consolidation (EWC).
Estimates diagonal Fisher Information Matrix on training data.
Penalizes changes to parameters critical to previously learned tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    """
    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0, device: torch.device = None):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device or next(model.parameters()).device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means: Dict[str, torch.Tensor] = {}
        self._fisher_matrices: Dict[str, torch.Tensor] = {}

    def compute_fisher(self, loader: DataLoader, num_samples: int = 200):
        """
        Estimates diagonal Fisher information using empirical gradients on training data.
        NOTE: Test data is NEVER used.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}

        samples_seen = 0
        for images, targets, _ in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            for i in range(images.size(0)):
                self.model.zero_grad()
                out = self.model(images[i:i+1])
                loss = nn.CrossEntropyLoss()(out, targets[i:i+1])
                loss.backward()

                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += (p.grad.data ** 2)

                samples_seen += 1
                if samples_seen >= num_samples:
                    break
            if samples_seen >= num_samples:
                break

        # Normalize by samples seen
        for n in fisher:
            fisher[n] /= max(1, samples_seen)

        # Store consolidated means and cumulative Fisher
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
            if n in self._fisher_matrices:
                self._fisher_matrices[n] += fisher[n]
            else:
                self._fisher_matrices[n] = fisher[n]

    def penalty(self) -> torch.Tensor:
        """
        Computes EWC quadratic loss penalty:
        L_ewc = sum_i (F_i * (theta_i - theta_i^*)^2)
        """
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.params.items():
            if n in self._fisher_matrices:
                loss += (self._fisher_matrices[n] * (p - self._means[n]) ** 2).sum()
        return self.ewc_lambda * loss
