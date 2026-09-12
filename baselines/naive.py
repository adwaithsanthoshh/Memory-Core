"""
Baseline 1: Naive Sequential Fine-Tuning (Mandatory Lower-Bound).
Trains incrementally on each incoming task without memory replay, EWC, or distillation.
Measures and records catastrophic forgetting across the complete accuracy matrix.
"""

import os
import time
import json
import torch
from typing import Dict, Any, Optional

from models.classifier import ContinualClassifier
from data.task_manager import TaskManager
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix, plot_forgetting_curves


def run_naive_baseline(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str = "results/naive"
) -> Dict[str, Any]:
    """
    Executes Naive Sequential Fine-Tuning across all tasks defined in task_manager.
    
    Returns:
        Dictionary of final metrics and artifact paths.
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
    print(f"STARTING BASELINE 1: NAIVE SEQUENTIAL FINE-TUNING")
    print(f"Total Tasks: {num_tasks} | Batch Size: {batch_size} | Epochs/Task: {epochs_per_task}")
    print(f"Mandatory Lower-Bound Reference")
    print(f"=======================================================")

    # Initialize model
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

    # Sequential Task Loop
    for task_id in range(num_tasks):
        print(f"\n>>> [Task {task_id + 1}/{num_tasks}] Training on Task {task_id} <<<")
        train_loader = task_manager.get_train_loader(task_id=task_id, batch_size=batch_size, shuffle=True)
        
        # 1. Train on current task only (no replay, no regularization)
        trainer.train_task_epochs(task_id=task_id, loader=train_loader, epochs=epochs_per_task)

        # 2. Evaluate on all tasks seen so far
        evaluator.evaluate_all_seen_tasks(
            model=trainer.model,
            current_task_idx=task_id,
            metrics_tracker=metrics_tracker,
            eval_batch_size=config.get("training", {}).get("eval_batch_size", 64)
        )

        # Save interim checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        trainer.save_checkpoint(ckpt_path, metadata={"task_id": task_id, "method": "naive"})

    total_training_time = time.time() - start_total_time

    # Compute final metrics & save artifacts
    additional_metrics = {
        "method": "naive",
        "training_time_seconds": round(total_training_time, 2),
        "config": config,
        "memory_budget": 0
    }
    metrics_tracker.save_artifacts(output_dir, additional_metrics=additional_metrics)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary.update(additional_metrics)

    # Save config.json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Generate plots
    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title="Naive Sequential Fine-Tuning — Accuracy Matrix (Catastrophic Forgetting)"
    )
    per_task_f, _ = metrics_tracker.compute_forgetting()
    plot_forgetting_curves(
        per_task_f,
        save_path=os.path.join(output_dir, "forgetting_curve.png"),
        title="Naive Fine-Tuning — Per-Task Forgetting"
    )

    print(f"\n=======================================================")
    print(f"NAIVE FINE-TUNING COMPLETED")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Average Forgetting:     {summary['average_forgetting']:.2f}%")
    print(f"Backward Transfer:      {summary['backward_transfer']:.2f}%")
    print(f"Artifacts saved in:     {output_dir}")
    print(f"=======================================================\n")

    return summary
