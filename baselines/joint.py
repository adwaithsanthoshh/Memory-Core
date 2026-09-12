"""
Baseline 2: Joint Training (Mandatory Upper-Bound Reference).
Trains a single model on all training data from all tasks simultaneously.
Serves as the non-continual upper-bound performance ceiling.
"""

import os
import time
import json
import torch
from typing import Dict, Any

from models.classifier import ContinualClassifier
from data.task_manager import TaskManager
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator, evaluate_loader
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix


def run_joint_baseline(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str = "results/joint"
) -> Dict[str, Any]:
    """
    Executes Joint Training on all tasks simultaneously.
    
    Returns:
        Summary metrics and artifact paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    start_total_time = time.time()

    num_tasks = task_manager.num_tasks
    epochs = config.get("training", {}).get("joint_epochs", 2)
    batch_size = config.get("training", {}).get("batch_size", 32)
    lr = config.get("training", {}).get("learning_rate", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    opt_type = config.get("training", {}).get("optimizer", "adamw")

    print(f"\n=======================================================")
    print(f"STARTING BASELINE 2: JOINT TRAINING (UPPER-BOUND REFERENCE)")
    print(f"Total Tasks Combined: {num_tasks} | Total Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Non-Continual Joint Upper Bound")
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

    # Get joint DataLoader containing all training samples
    joint_loader = task_manager.get_joint_train_loader(batch_size=batch_size, shuffle=True)
    print(f"Total Joint Training Samples: {len(joint_loader.dataset)}")

    # Train for specified epochs
    for epoch in range(epochs):
        print(f"\n>>> Joint Epoch {epoch + 1}/{epochs} <<<")
        trainer.train_task_epochs(task_id=0, loader=joint_loader, epochs=1)

    # Evaluate on every individual task's test set
    evaluator = ContinualEvaluator(task_manager=task_manager, device=device)
    metrics_tracker = ContinualMetricsTracker(num_tasks=num_tasks)

    print("\n--- Evaluating Joint Model Across All Tasks ---")
    per_task_accuracies = {}
    for j in range(num_tasks):
        loader = task_manager.get_task_test_loader(task_id=j, batch_size=config.get("training", {}).get("eval_batch_size", 64))
        acc = evaluate_loader(trainer.model, loader, device)
        # In joint training, all tasks are evaluated with the fully-trained joint model
        metrics_tracker.record_eval(trained_task_idx=num_tasks - 1, eval_task_idx=j, accuracy=acc)
        # Also fill diagonal for matrix consistency
        metrics_tracker.record_eval(trained_task_idx=j, eval_task_idx=j, accuracy=acc)
        per_task_accuracies[j] = acc
        print(f"  Task {j:02d} Accuracy: {acc:6.2f}%")

    full_bench_acc = evaluator.evaluate_full_benchmark(trainer.model)
    print(f"\n>>> Complete Benchmark Test Accuracy: {full_bench_acc:.2f}% <<<")

    total_time = time.time() - start_total_time

    # Save artifacts
    metrics_tracker.save_artifacts(output_dir)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary["method"] = "joint"
    summary["benchmark_accuracy"] = round(full_bench_acc, 2)
    summary["per_task_accuracies"] = {str(k): round(v, 2) for k, v in per_task_accuracies.items()}
    summary["training_time_seconds"] = round(total_time, 2)
    summary["config"] = config

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Plot joint evaluation
    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title="Joint Training — Upper Bound Accuracy"
    )

    # Save final model checkpoint
    trainer.save_checkpoint(os.path.join(output_dir, "joint_final_model.pt"), metadata={"method": "joint"})

    print(f"\n=======================================================")
    print(f"JOINT TRAINING (UPPER BOUND) COMPLETED")
    print(f"Average Final Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Full Test Accuracy:     {summary['benchmark_accuracy']:.2f}%")
    print(f"Artifacts saved in:     {output_dir}")
    print(f"=======================================================\n")

    return summary
