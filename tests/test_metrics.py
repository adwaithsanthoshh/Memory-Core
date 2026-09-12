"""
Unit tests for ContinualMetricsTracker.
Verifies hand-calculated accuracy matrix, average final accuracy, and average forgetting.
"""

import numpy as np
import pytest
from evaluation.metrics import ContinualMetricsTracker


def test_accuracy_matrix_and_metrics_calculation():
    # 3 tasks
    tracker = ContinualMetricsTracker(num_tasks=3)

    # Task 0 finished:
    # Eval on T0 = 90.0%
    tracker.record_eval(trained_task_idx=0, eval_task_idx=0, accuracy=90.0)
    assert tracker.get_average_accuracy(after_task_idx=0) == 90.0

    # Task 1 finished:
    # Eval on T0 = 60.0% (forgetting 30%), Eval on T1 = 80.0%
    tracker.record_eval(trained_task_idx=1, eval_task_idx=0, accuracy=60.0)
    tracker.record_eval(trained_task_idx=1, eval_task_idx=1, accuracy=80.0)
    assert tracker.get_average_accuracy(after_task_idx=1) == 70.0

    # Task 2 finished:
    # Eval on T0 = 40.0% (max was 90, so forgetting = 50%)
    # Eval on T1 = 50.0% (max was 80, so forgetting = 30%)
    # Eval on T2 = 85.0%
    tracker.record_eval(trained_task_idx=2, eval_task_idx=0, accuracy=40.0)
    tracker.record_eval(trained_task_idx=2, eval_task_idx=1, accuracy=50.0)
    tracker.record_eval(trained_task_idx=2, eval_task_idx=2, accuracy=85.0)

    # Average final accuracy: (40 + 50 + 85) / 3 = 175 / 3 = 58.33%
    avg_acc = tracker.get_average_accuracy(after_task_idx=2)
    assert round(avg_acc, 2) == 58.33

    # Forgetting:
    # F_0 = 90 - 40 = 50%
    # F_1 = 80 - 50 = 30%
    # Mean Forgetting = (50 + 30) / 2 = 40.0%
    per_task_f, avg_f = tracker.compute_forgetting(final_task_idx=2)
    assert per_task_f[0] == 50.0
    assert per_task_f[1] == 30.0
    assert avg_f == 40.0

    # Backward Transfer:
    # BWT_0 = 40 - 90 = -50%
    # BWT_1 = 50 - 80 = -30%
    # Mean BWT = -40.0%
    bwt = tracker.compute_backward_transfer(final_task_idx=2)
    assert bwt == -40.0
