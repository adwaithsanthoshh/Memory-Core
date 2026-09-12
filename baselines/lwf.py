"""
Baseline 5: Learning without Forgetting (LwF) (Li & Hoiem, 2017).
Uses knowledge distillation from a frozen copy of the previous task model to preserve old representations.
"""

import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from models.classifier import ContinualClassifier
from data.task_manager import TaskManager
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix


def run_lwf_baseline(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 1.0,
    output_dir: str = "results/lwf"
) -> Dict[str, Any]:
    """
    Executes Learning without Forgetting baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_total_time = time.time()

    num_tasks = task_manager.num_tasks
    epochs_per_task = config.get("training", {}).get("epochs_per_task", 1)
    batch_size = config.get("training", {}).get("batch_size", 32)
    lr = config.get("training", {}).get("learning_rate", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    opt_type = config.get("training", {}).get("optimizer", "adamw")

    print(f"\n=======================================================")
    print(f"STARTING BASELINE 5: LwF (TEMP = {temperature}, ALPHA = {alpha})")
    print(f"Tasks: {num_tasks}")
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
    previous_model: Optional[nn.Module] = None

    for task_id in range(num_tasks):
        print(f"\n>>> [Task {task_id + 1}/{num_tasks}] Training with LwF (Task {task_id}) <<<")
        train_loader = task_manager.get_train_loader(task_id=task_id, batch_size=batch_size, shuffle=True)

        def lwf_distillation_loss(cur_imgs, cur_targets, cur_logits):
            if previous_model is None:
                return torch.tensor(0.0, device=device)
            with torch.no_grad():
                prev_logits = previous_model(cur_imgs)
            
            # KD Loss using KL Divergence
            p_s = F.log_softmax(cur_logits / temperature, dim=1)
            p_t = F.softmax(prev_logits / temperature, dim=1)
            kd_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
            return alpha * kd_loss

        trainer.train_task_epochs(
            task_id=task_id,
            loader=train_loader,
            epochs=epochs_per_task,
            extra_loss_fn=lwf_distillation_loss if previous_model is not None else None
        )

        # Clone current model as frozen previous model for subsequent task
        previous_model = copy.deepcopy(trainer.model)
        previous_model.eval()
        for p in previous_model.parameters():
            p.requires_grad = False

        # Evaluate on all seen tasks
        evaluator.evaluate_all_seen_tasks(
            model=trainer.model,
            current_task_idx=task_id,
            metrics_tracker=metrics_tracker,
            eval_batch_size=config.get("training", {}).get("eval_batch_size", 64)
        )

        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        trainer.save_checkpoint(ckpt_path, metadata={"task_id": task_id, "method": "lwf"})

    total_time = time.time() - start_total_time

    metrics_tracker.save_artifacts(output_dir)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary["method"] = "lwf"
    summary["training_time_seconds"] = round(total_time, 2)
    summary["config"] = config

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title="LwF — Accuracy Matrix"
    )

    print(f"\n=======================================================")
    print(f"LwF COMPLETED")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Average Forgetting:     {summary['average_forgetting']:.2f}%")
    print(f"Backward Transfer:      {summary['backward_transfer']:.2f}%")
    print(f"=======================================================\n")

    return summary
