"""
Base Continual Trainer and Training Loop Utilities.
Provides standard SGD/AdamW optimization, logging, and checkpoint management.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger("memorycore.trainer")


class BaseContinualTrainer:
    """
    Standard training logic across tasks.
    Supports task-level iteration, checkpointing, and step-wise logging.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer_type: str = "adamw",
        momentum: float = 0.9
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = self._build_optimizer()
        self.training_logs: List[Dict] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine annealing scheduler: decays LR smoothly to avoid large gradient steps that overwrite old knowledge."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

    def train_task_epochs(
        self,
        task_id: int,
        loader: DataLoader,
        epochs: int = 1,
        extra_loss_fn = None
    ) -> float:
        """
        Trains model for specified number of epochs on current task data.
        
        Args:
            task_id: Task index.
            loader: Training DataLoader.
            epochs: Number of epochs to train.
            extra_loss_fn: Optional auxiliary loss callback for EWC, LwF, etc.
        """
        self.model.train()
        start_time = time.time()
        final_loss = 0.0

        # Re-build optimizer at the start of each task to flush stale momentum
        self.optimizer = self._build_optimizer()

        # Cosine scheduler decays LR gently over the task's epochs to reduce forgetting
        scheduler = self._build_scheduler(epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, targets, _) in enumerate(loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                if extra_loss_fn is not None:
                    aux_loss = extra_loss_fn(images, targets, outputs)
                    loss = loss + aux_loss

                loss.backward()
                # Gradient clipping: prevents catastrophic large parameter updates
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            scheduler.step()

            epoch_loss = total_loss / total if total > 0 else 0.0
            epoch_acc = (correct / total * 100.0) if total > 0 else 0.0
            final_loss = epoch_loss

            self.training_logs.append({
                "task_id": task_id,
                "epoch": epoch,
                "train_loss": round(epoch_loss, 4),
                "train_accuracy": round(epoch_acc, 2),
                "samples_processed": total,
                "elapsed_seconds": round(time.time() - start_time, 2)
            })

            print(f"Task {task_id} | Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        return final_loss

    def save_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict] = None):
        """Saves model weights and training state."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {}
        }, checkpoint_path)

    def save_training_logs(self, output_dir: str):
        """Exports step-by-step training log to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame(self.training_logs)
        df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)
