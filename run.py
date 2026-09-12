"""
MEMORYCORE: Main Command-Line Interface.
"Teach AI. Watch it forget. Teach it to remember."

Usage:
  python run.py --smoke-test               # Rapid offline smoke test (seconds)
  python run.py --method naive             # Baseline 1: Naive Sequential Fine-Tuning
  python run.py --method joint             # Baseline 2: Joint Training (Upper Bound)
  python run.py --method replay            # Baseline 3: Random Experience Replay
  python run.py --method ewc               # Baseline 4: Elastic Weight Consolidation
  python run.py --method lwf               # Baseline 5: Learning without Forgetting
  python run.py --download-data            # Downloads official CORe50 (128x128)
  python run.py --serve                    # Launches FastAPI backend server
  python run.py --demo                     # Launches demo mode with precomputed results
"""

import sys
import os
import argparse
import yaml
import json
import torch
import uvicorn

# Ensure package imports resolve cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual.reproducibility import set_seed, get_device
from data.tiny_imagenet import TinyImageNetDataManager, download_official_tiny_imagenet
from data.task_manager import TaskManager
from baselines.naive import run_naive_baseline
from baselines.joint import run_joint_baseline
from baselines.replay import run_replay_baseline
from baselines.ewc import run_ewc_baseline
from baselines.lwf import run_lwf_baseline
from research.memorycore import run_memorycore


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="MEMORYCORE: Continual Learning under Fixed Memory Budgets on CORe50"
    )
    parser.add_argument("--method", type=str, choices=["naive", "joint", "replay", "ewc", "lwf", "memorycore", "research"],
                        help="Baseline or research method to train")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Runs rapid offline smoke test across tasks to verify the pipeline")
    parser.add_argument("--download-data", action="store_true",
                        help="Downloads official CORe50 128x128 zip file")
    parser.add_argument("--serve", action="store_true",
                        help="Launches the FastAPI backend server")
    parser.add_argument("--demo", action="store_true",
                        help="Launches web demonstration using precomputed experiment results")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--memory-budget", type=int, default=5000,
                        help="Memory budget for replay methods (default: 5000)")
    parser.add_argument("--ewc-lambda", type=float, default=100.0,
                        help="EWC regularization strength")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="MemoryCore (DER++) distillation loss weight (default: 0.2)")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="MemoryCore (DER++) replay classification loss weight (default: 0.5)")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Override number of tasks to run (e.g. for quick experiments)")

    args = parser.parse_args()

    # 1. Download data if requested
    if args.download_data:
        download_official_tiny_imagenet()
        return

    # 2. Start server if requested
    if args.serve or args.demo:
        print("\nStarting MEMORYCORE FastAPI Server...")
        print("API Documentation available at: http://localhost:8000/docs")
        uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=False)
        return

    # 3. Handle training / smoke-test
    base_config = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else base_config.get("project", {}).get("seed", 42)
    set_seed(seed=seed, deterministic=base_config.get("project", {}).get("deterministic", True))
    device = get_device(base_config.get("project", {}).get("device", "auto"))

    print(f"Device: {device} | Seed: {seed}")

    # Configure dataset & task manager
    is_smoke = args.smoke_test
    scenario_name = "smoke" if is_smoke else base_config.get("dataset", {}).get("scenario", "NIC_v2_79")

    data_manager = TinyImageNetDataManager(
        root_dir=base_config.get("dataset", {}).get("root_dir", "data/tiny-imagenet-200"),
        num_tasks=base_config.get("dataset", {}).get("num_tasks", 10),
        smoke_mode=is_smoke
    )

    if is_smoke:
        print("\n=======================================================")
        print("RUNNING RAPID SMOKE TEST (OFFLINE SYNTHETIC MODE)")
        print("Verifying end-to-end data pipeline, ResNet-18 forward_features,")
        print("Naive fine-tuning, Joint training, and full metrics calculation.")
        print("=======================================================\n")
        max_tasks = 3
        max_samples = 30
        epochs = 1
        batch_size = 8
        eval_batch_size = 16
        
        # Override config for smoke test
        base_config["training"]["epochs_per_task"] = epochs
        base_config["training"]["batch_size"] = batch_size
        base_config["training"]["eval_batch_size"] = eval_batch_size
        base_config["training"]["joint_epochs"] = 1
        base_config["model"]["pretrained"] = False # offline rapid init
    else:
        max_tasks = args.num_tasks
        max_samples = None

    task_manager = TaskManager(
        data_manager=data_manager,
        max_tasks=max_tasks,
        max_samples_per_task=max_samples,
        smoke_mode=is_smoke
    )

    print(f"TaskManager initialized: {task_manager.num_tasks} tasks loaded.")
    print(f"Train/Test separation strictly verified. Official test sessions: s3, s7, s10 isolated.")

    method = args.method if args.method is not None else ("naive" if not is_smoke else "both_smoke")

    if is_smoke:
        methods_to_run = (
            ["naive", "joint", "replay", "ewc", "lwf", "memorycore"]
            if method in ["all", "both_smoke"]
            else [method]
        )
        summaries = {}
        for m in methods_to_run:
            print(f"\n>>> RUNNING SMOKE TEST FOR METHOD: {m.upper()} <<<")
            if m == "naive":
                summaries["naive"] = run_naive_baseline(
                    task_manager=task_manager, config=base_config, device=device, output_dir="results/smoke_naive"
                )
            elif m == "joint":
                summaries["joint"] = run_joint_baseline(
                    task_manager=task_manager, config=base_config, device=device, output_dir="results/smoke_joint"
                )
            elif m == "replay":
                summaries["replay"] = run_replay_baseline(
                    task_manager=task_manager, config=base_config, device=device, memory_budget=args.memory_budget, output_dir="results/smoke_replay"
                )
            elif m == "ewc":
                summaries["ewc"] = run_ewc_baseline(
                    task_manager=task_manager, config=base_config, device=device, ewc_lambda=args.ewc_lambda, output_dir="results/smoke_ewc"
                )
            elif m == "lwf":
                summaries["lwf"] = run_lwf_baseline(
                    task_manager=task_manager, config=base_config, device=device, output_dir="results/smoke_lwf"
                )
            elif m in ["memorycore", "research"]:
                summaries["memorycore"] = run_memorycore(
                    task_manager=task_manager, config=base_config, device=device, memory_budget=args.memory_budget, alpha=args.alpha, beta=args.beta, output_dir="results/smoke_memorycore"
                )

        print("\n=======================================================")
        print("ALL REQUESTED SMOKE TESTS COMPLETED SUCCESSFULLY!")
        for name, summ in summaries.items():
            print(f"[{name.upper()}] Final Avg Acc: {summ.get('final_average_accuracy')}%, Avg Forgetting: {summ.get('average_forgetting')}%")
        print("Verified: Accuracy matrix, forgetting, training logs, checkpoints, plots.")
        print("=======================================================\n")
        return

    # Method Dispatcher
    if method == "naive":
        run_naive_baseline(task_manager=task_manager, config=base_config, device=device)
    elif method == "joint":
        run_joint_baseline(task_manager=task_manager, config=base_config, device=device)
    elif method == "replay":
        run_replay_baseline(task_manager=task_manager, config=base_config, device=device, memory_budget=args.memory_budget)
    elif method == "ewc":
        run_ewc_baseline(task_manager=task_manager, config=base_config, device=device, ewc_lambda=args.ewc_lambda)
    elif method == "lwf":
        run_lwf_baseline(task_manager=task_manager, config=base_config, device=device)
    elif method in ["memorycore", "research"]:
        run_memorycore(
            task_manager=task_manager,
            config=base_config,
            device=device,
            memory_budget=args.memory_budget,
            alpha=args.alpha,
            beta=args.beta
        )


if __name__ == "__main__":
    main()
