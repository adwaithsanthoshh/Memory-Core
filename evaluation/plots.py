"""
Plotting utilities for Continual Learning evaluation.
Generates:
1. Accuracy Matrix Heatmap
2. Average Accuracy vs Task Curve
3. Forgetting Dynamics Chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_accuracy_matrix(accuracy_matrix: np.ndarray, save_path: str, title: str = "Accuracy Matrix"):
    """
    Plots and saves the lower-triangular accuracy matrix heatmap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    masked_matrix = np.ma.masked_invalid(accuracy_matrix)
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')

    cax = ax.imshow(masked_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    fig.colorbar(cax, label="Accuracy (%)")

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Evaluation Task $j$", fontsize=12)
    ax.set_ylabel("After Training Task $i$", fontsize=12)

    # Annotate if small matrix (<= 15 tasks)
    n = accuracy_matrix.shape[0]
    if n <= 15:
        for i in range(n):
            for j in range(n):
                val = accuracy_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", 
                            color="white" if val < 50 else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_forgetting_curves(forgetting_dict: dict, save_path: str, title: str = "Per-Task Forgetting"):
    """
    Plots per-task forgetting bar chart.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tasks = sorted(list(forgetting_dict.keys()))
    vals = [forgetting_dict[t] for t in tasks]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([f"T{t}" for t in tasks], vals, color="#d9534f")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Task ID", fontsize=12)
    ax.set_ylabel("Accuracy Drop (%)", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
