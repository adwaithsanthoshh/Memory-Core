"""
Baseline 3: Random Experience Replay.
Replays a fixed budget of stored historical exemplars (e.g. 50, 100, 250, 500, 1000)
using uniform reservoir sampling during sequential learning.
"""

import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.classifier import ContinualClassifier
from data.task_manager import TaskManager
from continual.memory import ReplayMemoryBuffer
from continual.trainer import BaseContinualTrainer
from continual.evaluator import ContinualEvaluator
from evaluation.metrics import ContinualMetricsTracker
from evaluation.plots import plot_accuracy_matrix, plot_forgetting_curves


def run_replay_baseline(
    task_manager: TaskManager,
    config: Dict[str, Any],
    device: torch.device,
    memory_budget: int = 250,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes Experience Replay baseline under a fixed memory budget.
    """
    if output_dir is None:
        output_dir = f"results/replay_m{memory_budget}"
        
    os.makedirs(output_dir, exist_ok=True)
    start_total_time = time.time()

    num_tasks = task_manager.num_tasks
    epochs_per_task = config.get("training", {}).get("epochs_per_task", 1)
    batch_size = config.get("training", {}).get("batch_size", 32)
    lr = config.get("training", {}).get("learning_rate", 0.001)
    weight_decay = config.get("training", {}).get("weight_decay", 0.0001)
    opt_type = config.get("training", {}).get("optimizer", "adamw")

    print(f"\n=======================================================")
    print(f"STARTING BASELINE 3: EXPERIENCE REPLAY (BUDGET = {memory_budget})")
    print(f"Tasks: {num_tasks} | Memory Budget: {memory_budget} samples")
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
    memory_buffer = ReplayMemoryBuffer(capacity=memory_budget)

    # Sequential loop
    for task_id in range(num_tasks):
        print(f"\n>>> [Task {task_id + 1}/{num_tasks}] Training with Replay (Mem: {len(memory_buffer)}/{memory_budget}) <<<")
        train_loader = task_manager.get_train_loader(task_id=task_id, batch_size=batch_size, shuffle=True)

        # Replay loss callback
        def replay_loss_fn(cur_imgs, cur_targets, cur_logits):
            sample = memory_buffer.sample_batch(batch_size=min(batch_size, len(memory_buffer)))
            if sample is None:
                return torch.tensor(0.0, device=device)
            m_imgs, m_targets, _ = sample
            m_imgs = m_imgs.to(device)
            m_targets = m_targets.to(device)
            m_logits = trainer.model(m_imgs)
            return nn.CrossEntropyLoss()(m_logits, m_targets)

        # Train on current task + replayed memory
        trainer.train_task_epochs(
            task_id=task_id,
            loader=train_loader,
            epochs=epochs_per_task,
            extra_loss_fn=replay_loss_fn if len(memory_buffer) > 0 else None
        )

        # Update memory buffer with current task data
        task_ds = task_manager.get_train_dataset(task_id)
        samples_per_task_quota = max(1, memory_budget // (task_id + 1))
        memory_buffer.add_from_task(task_ds, num_to_add=samples_per_task_quota)

        # Evaluate on all seen tasks
        evaluator.evaluate_all_seen_tasks(
            model=trainer.model,
            current_task_idx=task_id,
            metrics_tracker=metrics_tracker,
            eval_batch_size=config.get("training", {}).get("eval_batch_size", 64)
        )

        # Checkpoint
        ckpt_path = os.path.join(output_dir, f"checkpoint_task_{task_id}.pt")
        trainer.save_checkpoint(ckpt_path, metadata={"task_id": task_id, "memory_budget": memory_budget})

    total_time = time.time() - start_total_time

    # Save artifacts
    metrics_tracker.save_artifacts(output_dir)
    trainer.save_training_logs(output_dir)

    summary = metrics_tracker.get_summary_metrics()
    summary["method"] = "replay"
    summary["memory_budget"] = memory_budget
    summary["training_time_seconds"] = round(total_time, 2)
    summary["config"] = config

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    plot_accuracy_matrix(
        metrics_tracker.accuracy_matrix,
        save_path=os.path.join(output_dir, "accuracy_matrix.png"),
        title=f"Experience Replay (Budget = {memory_budget}) — Accuracy Matrix"
    )

    print(f"\n=======================================================")
    print(f"EXPERIENCE REPLAY (BUDGET {memory_budget}) COMPLETED")
    print(f"Final Average Accuracy: {summary['final_average_accuracy']:.2f}%")
    print(f"Average Forgetting:     {summary['average_forgetting']:.2f}%")
    print(f"Backward Transfer:      {summary['backward_transfer']:.2f}%")
    print(f"=======================================================\n")

    return summary
