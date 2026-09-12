"""
Baseline 4: Elastic Weight Consolidation (EWC).
Estimates Fisher information matrices on training data to protect consolidated weights.
Mandatory rule: Never use test data to compute Fisher information.
"""

import os
import time
import json
import torch
from typing import Dict, Any, Optional

from models.classifier import ContinualClassifier
from models.ewc import EWC
from data.task_manager import TaskManager
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix, plot_forgetting_curves


def run_ewc_baseline(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    ewc_lambda: float = 100.0,
    output_dir: str = "results/ewc"
) -> Dict[str, Any]:
    """
    Executes EWC baseline.
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
    print(f"STARTING BASELINE 4: EWC (LAMBDA = {ewc_lambda})")
    print(f"Tasks: {num_tasks} | Fisher Samples: 200")
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
    ewc = EWC(model=trainer.model, ewc_lambda=ewc_lambda, device=device)

    for task_id in range(num_tasks):
        print(f"\n>>> [Task {task_id + 1}/{num_tasks}] Training with EWC (Task {task_id}) <<<")
        train_loader = task_manager.get_train_loader(task_id=task_id, batch_size=batch_size, shuffle=True)

        def ewc_loss_fn(cur_imgs, cur_targets, cur_logits):
            if task_id == 0:
                return torch.tensor(0.0, device=device)
            return ewc.penalty()

        trainer.train_task_epochs(
            task_id=task_id,
            loader=train_loader,
            epochs=epochs_per_task,
            extra_loss_fn=ewc_loss_fn if task_id > 0 else None
        )

        # Estimate Fisher information using TRAINING data only (never test data)
        fisher_loader = task_manager.get_train_loader(task_id=task_id, batch_size=16, shuffle=False)
        ewc.compute_fisher(fisher_loader, num_samples=config.get("continual", {}).get("ewc_fisher_samples", 200))

        # Evaluate on all seen tasks
        evaluator.evaluate_all_seen_tasks(
            model=trainer.model,
            current_task_idx=task_id,
            metrics_tracker=metrics_tracker,
            eval_batch_size=config.get("training", {}).get("eval_batch_size", 64)
        )

        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        trainer.save_checkpoint(ckpt_path, metadata={"task_id": task_id, "method": "ewc", "lambda": ewc_lambda})

    total_time = time.time() - start_total_time

    # Save artifacts
    metrics_tracker.save_artifacts(output_dir)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary["method"] = "ewc"
    summary["ewc_lambda"] = ewc_lambda
    summary["training_time_seconds"] = round(total_time, 2)
    summary["config"] = config

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title=f"EWC (Lambda = {ewc_lambda}) — Accuracy Matrix"
    )

    print(f"\n=======================================================")
    print(f"EWC COMPLETED")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Average Forgetting:     {summary['average_forgetting']:.2f}%")
    print(f"Backward Transfer:      {summary['backward_transfer']:.2f}%")
    print(f"=======================================================\n")

    return summary
