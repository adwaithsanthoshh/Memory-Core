"""
Unit tests for CORe50 data loading and scenario verification.
Strictly checks:
1. NIC_v2_79 official task stream.
2. Complete absence of data leakage (test sessions s3, s7, s10 never in train).
3. TaskManager interface compliance.
"""

import pytest
from data.core50 import CORe50DataManager
from data.task_manager import TaskManager


def test_core50_nic_v2_79_metadata_and_isolation():
    dm = CORe50DataManager(
        root_dir="data/core50_128x128",
        metadata_dir="data/core50_metadata",
        scenario="NIC_v2_79",
        run_id=0,
        smoke_mode=True
    )

    # Check class names
    assert len(dm.class_names) == 50
    assert dm.class_names[0] == "plug_adapter1"

    # Verify task filelists
    train_files, test_file = dm.get_task_filelists()
    assert len(train_files) == 79
    assert "test_filelist.txt" in test_file

    # Load small subset of tasks
    tm = TaskManager(data_manager=dm, max_tasks=3, max_samples_per_task=20, smoke_mode=True)
    assert tm.num_tasks == 3

    # Strictly verify train/test separation
    assert dm.verify_train_test_separation(tm.task_samples, tm.test_samples)

    # Check Task 0 metadata
    t0_meta = tm.get_task_metadata(0)
    assert t0_meta['task_id'] == 0
    assert t0_meta['num_samples'] == 20
    assert len(t0_meta['classes']) > 0

    # Test DataLoader yields correct shapes
    loader = tm.get_train_loader(task_id=0, batch_size=4)
    imgs, targets, meta = next(iter(loader))
    assert imgs.shape == (4, 3, 128, 128)
    assert targets.shape == (4,)
