"""
Continual Evaluator for MEMORYCORE.
Strictly evaluates models across all previous tasks up to the current task.
Populates the Accuracy Matrix and logs progress.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging

from data.task_manager import TaskManager
from evaluation.metrics import ContinualMetricsTracker

logger = logging.getLogger("memorycore.evaluator")


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Computes top-1 classification accuracy on a given DataLoader.
    
    Returns:
        Accuracy in percentage [0.0, 100.0].
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images, targets, _ = batch
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return accuracy


def evaluate_loader_task_il(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_classes: list
) -> float:
    """
    Task-Incremental (Task-IL) evaluation protocol.
    Restricts predictions to only the classes that belong to this specific task.
    This eliminates interference from neurons of unseen/other tasks,
    giving a fair measure of the model's retained knowledge on that task.

    Returns:
        Accuracy in percentage [0.0, 100.0].
    """
    model.eval()
    correct = 0
    total = 0

    # Create a mask tensor — only task_classes indices are valid predictions
    task_class_tensor = torch.tensor(task_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in loader:
            images, targets, _ = batch
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)  # shape: [B, num_classes]

            # Gather only the logits for this task's classes
            task_logits = outputs[:, task_class_tensor]  # shape: [B, len(task_classes)]

            # Map prediction indices back to global class labels
            local_preds = task_logits.argmax(dim=1)  # index within task_classes
            global_preds = task_class_tensor[local_preds]  # global class label

            correct += (global_preds == targets).sum().item()
            total += targets.size(0)

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return accuracy


class ContinualEvaluator:
    """
    Orchestrates per-task evaluation at each stage of the continual learning stream.
    """
    def __init__(self, task_manager: TaskManager, device: torch.device):
        self.tm = task_manager
        self.device = device

    def evaluate_all_seen_tasks(
        self,
        model: nn.Module,
        current_task_idx: int,
        metrics_tracker: ContinualMetricsTracker,
        eval_batch_size: int = 64
    ) -> Dict[int, float]:
        """
        Evaluates the model on every task j in [0, ..., current_task_idx].
        Uses Task-IL protocol: predictions are restricted to each task's own class set,
        eliminating cross-task neuron interference for fair evaluation.
        Records results directly in the metrics_tracker accuracy matrix.
        """
        model.eval()
        task_accuracies = {}

        print(f"\n--- Evaluating after Task {current_task_idx} (Task-IL, tasks 0..{current_task_idx}) ---")
        for j in range(current_task_idx + 1):
            loader = self.tm.get_task_test_loader(task_id=j, batch_size=eval_batch_size)
            task_classes = self.tm.get_task_metadata(j)['classes']
            acc = evaluate_loader_task_il(model, loader, self.device, task_classes)
            metrics_tracker.record_eval(trained_task_idx=current_task_idx, eval_task_idx=j, accuracy=acc)
            task_accuracies[j] = acc
            print(f"  Eval on Task {j:02d}: {acc:6.2f}%")

        avg_acc = metrics_tracker.get_average_accuracy(after_task_idx=current_task_idx)
        print(f"  => Current Average Accuracy (Tasks 0..{current_task_idx}): {avg_acc:.2f}%")
        return task_accuracies

    def evaluate_full_benchmark(self, model: nn.Module, eval_batch_size: int = 64) -> float:
        """
        Evaluates model on the complete official test benchmark (all 50 classes).
        """
        loader = self.tm.get_full_test_loader(batch_size=eval_batch_size)
        return evaluate_loader(model, loader, self.device)
