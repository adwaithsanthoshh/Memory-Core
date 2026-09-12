"""
Forgetting analysis utilities.
Decomposes catastrophic forgetting across classes, sessions, and task distances.
"""

from typing import Dict, List, Tuple
import numpy as np


def analyze_forgetting_dynamics(accuracy_matrix: np.ndarray) -> Dict:
    """
    Analyzes temporal decay patterns in the accuracy matrix.
    
    Args:
        accuracy_matrix: 2D array where R[i, j] is acc on task j after training task i.
    """
    num_tasks = accuracy_matrix.shape[0]
    initial_accs = [accuracy_matrix[i, i] for i in range(num_tasks) if not np.isnan(accuracy_matrix[i, i])]
    final_accs = [accuracy_matrix[num_tasks - 1, j] for j in range(num_tasks) if not np.isnan(accuracy_matrix[num_tasks - 1, j])]
    
    retention_rates = []
    for init, fin in zip(initial_accs, final_accs):
        if init > 0:
            retention_rates.append(fin / init)

    return {
        "mean_initial_accuracy": float(np.mean(initial_accs)) if initial_accs else 0.0,
        "mean_final_accuracy": float(np.mean(final_accs)) if final_accs else 0.0,
        "mean_knowledge_retention_rate": float(np.mean(retention_rates)) if retention_rates else 0.0
    }
