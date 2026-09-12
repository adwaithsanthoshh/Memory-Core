"""
Evaluation Metrics for Continual Learning.
Implements:
1. Complete Accuracy Matrix R (where R[i, j] is eval accuracy on task j after training task i)
2. Average Final Accuracy
3. Average Forgetting (F_j = max_{k < T} R[k, j] - R[T-1, j])
4. Backward Transfer (BWT)
5. Comprehensive CSV / JSON serialization.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import json
import os


class ContinualMetricsTracker:
    """
    Tracks and computes all continual learning metrics across sequential tasks.
    Guarantees strict mathematical compliance with Hackathon requirements.
    """
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        # R[i, j]: accuracy on task j after completing task i
        # Initialized with NaN
        self.accuracy_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
        self.current_completed_task = -1

    def record_eval(self, trained_task_idx: int, eval_task_idx: int, accuracy: float):
        """
        Records evaluation accuracy of model after training on trained_task_idx
        when tested on eval_task_idx. Accuracy should be in range [0.0, 100.0] or [0.0, 1.0].
        Normalizes to percentage [0.0, 100.0].
        """
        if accuracy <= 1.0 and accuracy > 0.0:
            accuracy = accuracy * 100.0
            
        self.accuracy_matrix[trained_task_idx, eval_task_idx] = float(accuracy)
        if trained_task_idx > self.current_completed_task:
            self.current_completed_task = trained_task_idx

    def get_average_accuracy(self, after_task_idx: Optional[int] = None) -> float:
        """
        Computes Average Accuracy over all tasks seen so far up to after_task_idx:
        A_i = (1 / (i + 1)) * sum_{j=0}^{i} R[i, j]
        """
        if after_task_idx is None:
            after_task_idx = self.current_completed_task
            
        if after_task_idx < 0:
            return 0.0

        row = self.accuracy_matrix[after_task_idx, :after_task_idx + 1]
        valid = row[~np.isnan(row)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def compute_forgetting(self, final_task_idx: Optional[int] = None) -> Tuple[Dict[int, float], float]:
        """
        Computes per-task forgetting and Average Forgetting.
        
        For each task j < final_task_idx:
            F_j = max_{k in [j, final_task_idx - 1]} R[k, j] - R[final_task_idx, j]
            
        Average Forgetting = mean_{j=0}^{final_task_idx - 1}(F_j)
        """
        if final_task_idx is None:
            final_task_idx = self.current_completed_task

        if final_task_idx <= 0:
            return {}, 0.0

        per_task_forgetting = {}
        forgetting_vals = []

        for j in range(final_task_idx):
            # Accuracies on task j from task j up to final_task_idx - 1
            historical_accs = self.accuracy_matrix[j:final_task_idx, j]
            final_acc = self.accuracy_matrix[final_task_idx, j]

            if not np.isnan(final_acc) and len(historical_accs) > 0:
                max_prior = np.nanmax(historical_accs)
                f_j = float(max_prior - final_acc)
                per_task_forgetting[j] = f_j
                forgetting_vals.append(f_j)

        avg_forgetting = float(np.mean(forgetting_vals)) if len(forgetting_vals) > 0 else 0.0
        return per_task_forgetting, avg_forgetting

    def compute_backward_transfer(self, final_task_idx: Optional[int] = None) -> float:
        """
        Computes Backward Transfer (BWT):
        BWT = (1 / (T - 1)) * sum_{j=0}^{T - 2} (R[T - 1, j] - R[j, j])
        """
        if final_task_idx is None:
            final_task_idx = self.current_completed_task

        if final_task_idx <= 0:
            return 0.0

        bwt_vals = []
        for j in range(final_task_idx):
            init_acc = self.accuracy_matrix[j, j]
            final_acc = self.accuracy_matrix[final_task_idx, j]
            if not np.isnan(init_acc) and not np.isnan(final_acc):
                bwt_vals.append(final_acc - init_acc)

        return float(np.mean(bwt_vals)) if len(bwt_vals) > 0 else 0.0

    def get_summary_metrics(self) -> Dict[str, float]:
        """Returns consolidated metric dictionary."""
        per_task_f, avg_f = self.compute_forgetting()
        avg_acc = self.get_average_accuracy()
        bwt = self.compute_backward_transfer()
        
        return {
            "final_average_accuracy": round(avg_acc, 2),
            "average_forgetting": round(avg_f, 2),
            "backward_transfer": round(bwt, 2),
            "tasks_completed": self.current_completed_task + 1,
            "total_tasks": self.num_tasks
        }

    def save_artifacts(self, output_dir: str, additional_metrics: Optional[Dict] = None):
        """
        Saves:
        1. accuracy_matrix.csv
        2. forgetting.csv
        3. metrics.json
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Accuracy Matrix CSV
        t_count = self.current_completed_task + 1
        sub_matrix = self.accuracy_matrix[:t_count, :t_count]
        columns = [f"Eval_Task_{j}" for j in range(t_count)]
        indices = [f"After_Task_{i}" for i in range(t_count)]
        df_matrix = pd.DataFrame(sub_matrix, index=indices, columns=columns)
        matrix_path = os.path.join(output_dir, "accuracy_matrix.csv")
        df_matrix.to_csv(matrix_path)

        # 2. Forgetting CSV
        per_task_f, avg_f = self.compute_forgetting()
        f_records = []
        for t_idx, f_val in per_task_f.items():
            f_records.append({"task_id": t_idx, "forgetting": round(f_val, 2)})
        df_forgetting = pd.DataFrame(f_records)
        df_forgetting.to_csv(os.path.join(output_dir, "forgetting.csv"), index=False)

        # 3. Summary Metrics JSON
        summary = self.get_summary_metrics()
        summary["per_task_forgetting"] = {str(k): round(v, 2) for k, v in per_task_f.items()}
        
        if additional_metrics:
            summary.update(additional_metrics)
            
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return matrix_path
