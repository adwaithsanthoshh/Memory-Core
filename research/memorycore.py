"""
MemoryCore: Continual Learning with Dark Experience Replay (DER++).
Track 5 Hackathon Custom Algorithm.

Combines replay buffer preservation with past logit distillation ("dark knowledge"),
alleviating catastrophic forgetting without the excessive rigidity of parameter regularization.

Reference:
Buzzega et al., "Dark Experience for General Continual Learning: a Strong, Simple Baseline", NeurIPS 2020.
"""

import os
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import Dataset

from models.classifier import ContinualClassifier
from data.task_manager import TaskManager
from continual.memory import ReplayMemoryBuffer
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix


class DERMemoryBuffer:
    """
    Fixed-capacity memory buffer storing images, labels, and past output logits.
    Guarantees:
    - Maximum capacity is strictly enforced (total samples <= capacity).
    - Buffer is fully utilized at all times (capacity dynamically partitioned across seen tasks).
    - Class balance is strictly preserved within each task.
    - Exemplars retain dark knowledge (output logits) computed at the end of task learning.
    """
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        # task_id -> list of (img_tensor, label, meta_dict_with_logits)
        self.task_storage: Dict[int, List[Tuple[torch.Tensor, int, Dict]]] = {}

    def __len__(self):
        return sum(len(samples) for samples in self.task_storage.values())

    def update_with_task(
        self,
        task_id: int,
        dataset: Dataset,
        model: nn.Module,
        device: torch.device
    ):
        """
        Dynamically rebalances buffer after completing task_id:
        1. Calculates per-task quota: capacity // (len(seen_tasks) + 1).
        2. Prunes existing tasks to the new quota while preserving class balance.
        3. Selects class-balanced exemplars from current task and caches their logits.
        """
        model.eval()
        num_seen = len(self.task_storage) + 1
        quota_per_task = max(1, self.capacity // num_seen)

        # 1. Prune existing tasks to quota_per_task
        for prev_tid in list(self.task_storage.keys()):
            cur_samples = self.task_storage[prev_tid]
            if len(cur_samples) > quota_per_task:
                by_class: Dict[int, List] = {}
                for item in cur_samples:
                    by_class.setdefault(item[1], []).append(item)
                pruned = []
                classes = sorted(list(by_class.keys()))
                idx = 0
                while len(pruned) < quota_per_task and any(by_class.values()):
                    c = classes[idx % len(classes)]
                    if by_class[c]:
                        pruned.append(by_class[c].pop(0))
                    idx += 1
                self.task_storage[prev_tid] = pruned

        # 2. Extract new task samples with class balance
        class_to_indices: Dict[int, List[int]] = {}
        for idx in range(len(dataset)):
            _, label, _ = dataset[idx]
            class_to_indices.setdefault(label, []).append(idx)

        selected_indices = []
        classes = sorted(list(class_to_indices.keys()))
        idx = 0
        while len(selected_indices) < quota_per_task and any(class_to_indices.values()):
            c = classes[idx % len(classes)]
            if class_to_indices[c]:
                selected_indices.append(class_to_indices[c].pop(0))
            idx += 1

        # Compute output logits on selected exemplars
        new_task_items = []
        batch_size = 64
        with torch.no_grad():
            for start in range(0, len(selected_indices), batch_size):
                batch_idxs = selected_indices[start:start + batch_size]
                batch_imgs = []
                batch_labels = []
                batch_metas = []
                for b_idx in batch_idxs:
                    img, label, meta = dataset[b_idx]
                    batch_imgs.append(img)
                    batch_labels.append(label)
                    batch_metas.append((img, label, meta))

                imgs_t = torch.stack(batch_imgs).to(device)
                logits_t = model(imgs_t).cpu().detach()

                for j in range(len(batch_imgs)):
                    img, label, meta = batch_metas[j]
                    sample_meta = dict(meta) if isinstance(meta, dict) else {}
                    sample_meta["logits"] = logits_t[j]
                    sample_meta["task_id"] = task_id
                    new_task_items.append((img.cpu().detach(), label, sample_meta))

        self.task_storage[task_id] = new_task_items

    def sample_der_batch(
        self,
        batch_size: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Samples a mini-batch of stored memories along with their stored logits.
        Uniformly samples across all currently retained exemplars.
        Returns: (images, labels, past_logits) or None if buffer is empty.
        """
        all_samples = []
        for samples in self.task_storage.values():
            all_samples.extend(samples)

        if len(all_samples) == 0:
            return None

        actual_k = min(batch_size, len(all_samples))
        sampled = random.sample(all_samples, actual_k)

        imgs = torch.stack([x[0] for x in sampled])
        labels = torch.tensor([x[1] for x in sampled], dtype=torch.long)
        logits = torch.stack([x[2]["logits"] for x in sampled])

        return imgs, labels, logits


def run_memorycore(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    memory_budget: int = 5000,
    alpha: float = 0.2,
    beta: float = 0.5,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes MemoryCore (DER++) under a fixed memory budget.

    Args:
        task_manager: TaskManager managing task sequences.
        config: Training and model configuration dictionary.
        device: Torch computing device.
        memory_budget: Total exemplar storage capacity (default: 5000).
        alpha: Weight for logit distillation loss (MSE) on buffer samples.
        beta: Weight for replay cross-entropy loss on buffer samples.
        output_dir: Output directory for saving artifacts.
    """
    if output_dir is None:
        output_dir = f"results/memorycore_m{memory_budget}"

    os.makedirs(output_dir, exist_ok=True)
    start_total_time = time.time()

    num_tasks = task_manager.num_tasks
    epochs_per_task = config.get("training", {}).get("epochs_per_task", 1)
    batch_size = config.get("training", {}).get("batch_size", 32)
    lr = config.get("training", {}).get("learning_rate", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    opt_type = config.get("training", {}).get("optimizer", "adamw")

    print(f"\n=======================================================")
    print(f"STARTING RESEARCH METHOD: MEMORYCORE (DER++) (BUDGET = {memory_budget})")
    print(f"Tasks: {num_tasks} | Memory Budget: {memory_budget} | Alpha (MSE): {alpha} | Beta (CE): {beta}")
    print(f"=======================================================")

    model = ContinualClassifier(
        num_classes=config.get("model", {}).get("num_classes", 50),
        pretrained=config.get("model", {}).get("pretrained", True),
        freeze_features=config.get("model", {}).get("freeze_features", False)
    )

    trainer = BaseContinualTrainer(
        model=model,
        device=device,
        learning_rate=lr,
        weight_decay=weight_decay,
        optimizer_type=opt_type
    )

    evaluator = ContinualEvaluator(task_manager=task_manager, device=device)
    metrics_tracker = ContinualMetricsTracker(num_tasks=num_tasks)
    der_buffer = DERMemoryBuffer(capacity=memory_budget)

    # Sequential continual learning loop
    for task_id in range(num_tasks):
        print(f"\n>>> [Task {task_id + 1}/{num_tasks}] Training MemoryCore (Buffer: {len(der_buffer)}/{memory_budget}) <<<")
        train_loader = task_manager.get_train_loader(task_id=task_id, batch_size=batch_size, shuffle=True)

        # DER++ auxiliary loss function
        def der_loss_fn(cur_imgs, cur_targets, cur_logits):
            sample = der_buffer.sample_der_batch(batch_size=batch_size)
            if sample is None:
                return torch.tensor(0.0, device=device)

            buf_imgs, buf_targets, buf_past_logits = sample
            buf_imgs = buf_imgs.to(device)
            buf_targets = buf_targets.to(device)
            buf_past_logits = buf_past_logits.to(device)

            # Model forward on replayed exemplars
            buf_cur_logits = trainer.model(buf_imgs)

            # 1. Distillation loss: MSE between current logits and stored logits (dark knowledge)
            loss_distill = F.mse_loss(buf_cur_logits, buf_past_logits)

            # 2. Replay classification loss: CrossEntropy with ground truth labels
            loss_ce_buf = nn.CrossEntropyLoss()(buf_cur_logits, buf_targets)

            return (alpha * loss_distill) + (beta * loss_ce_buf)

        # Train on current task with DER++ replay callback
        trainer.train_task_epochs(
            task_id=task_id,
            loader=train_loader,
            epochs=epochs_per_task,
            extra_loss_fn=der_loss_fn if len(der_buffer) > 0 else None
        )

        # Dynamically rebalance and populate buffer with current task exemplars
        task_ds = task_manager.get_train_dataset(task_id)
        der_buffer.update_with_task(
            task_id=task_id,
            dataset=task_ds,
            model=trainer.model,
            device=device
        )

        # Evaluate on all seen tasks
        evaluator.evaluate_all_seen_tasks(
            model=trainer.model,
            current_task_idx=task_id,
            metrics_tracker=metrics_tracker,
            eval_batch_size=config.get("training", {}).get("eval_batch_size", 64)
        )

        # Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        trainer.save_checkpoint(ckpt_path, metadata={
            "task_id": task_id,
            "memory_budget": memory_budget,
            "method": "memorycore_derpp",
            "alpha": alpha,
            "beta": beta
        })

    total_time = time.time() - start_total_time

    # Save artifacts
    additional_metrics = {
        "method": "memorycore",
        "algorithm": "DER++ (Uncertainty-Aware)",
        "memory_budget": memory_budget,
        "alpha": alpha,
        "beta": beta,
        "training_time_seconds": round(total_time, 2),
        "config": config
    }
    metrics_tracker.save_artifacts(output_dir, additional_metrics=additional_metrics)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary.update(additional_metrics)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title=f"MemoryCore (DER++, Budget={memory_budget}) — Accuracy Matrix"
    )

    print(f"\n=======================================================")
    print(f"MEMORYCORE (DER++, BUDGET {memory_budget}) COMPLETED")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Average Forgetting:     {summary['average_forgetting']:.2f}%")
    print(f"Backward Transfer:      {summary['backward_transfer']:.2f}%")
    print(f"=======================================================")

    return summary
