"""
TaskManager: Provides task-stream abstraction for Continual Learning on CORe50.
Exposes task_id, sample indices, labels, session metadata, class/object IDs,
and clean train/test membership.
"""

from typing import List, Dict, Set, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from data.tiny_imagenet import TinyImageNetDataManager, TinyImageNetDataset
from data.transforms import get_transforms


class TaskManager:
    """
    Manages sequential task streams according to Split Tiny-ImageNet benchmarks.
    Guarantees strict train/test isolation, deterministic batch ordering,
    and granular inspection of seen/new classes and sessions per task.
    """
    def __init__(
        self,
        data_manager: TinyImageNetDataManager,
        max_tasks: Optional[int] = None,
        max_samples_per_task: Optional[int] = None,
        smoke_mode: bool = False
    ):
        self.dm = data_manager
        self.smoke_mode = smoke_mode
        self.max_tasks = max_tasks
        self.max_samples_per_task = max_samples_per_task

        # Load task samples and test set
        self.task_samples, self.test_samples = self.dm.load_scenario(
            max_tasks=max_tasks,
            max_samples_per_task=max_samples_per_task
        )
        self.num_tasks = len(self.task_samples)

        # Strictly verify train/test isolation
        self.dm.verify_train_test_separation(self.task_samples, self.test_samples)

        # Build class-to-test-samples index for fast per-task evaluation
        self._test_samples_by_class: Dict[int, List[Dict]] = {}
        for s in self.test_samples:
            lbl = s['label']
            if lbl not in self._test_samples_by_class:
                self._test_samples_by_class[lbl] = []
            self._test_samples_by_class[lbl].append(s)

        # Build cumulative history tracking
        self.seen_classes: Set[int] = set()
        self.task_metadata: List[Dict] = []
        self._build_task_metadata()

    def _build_task_metadata(self):
        """Analyzes each task in sequence to track class emergence and statistics."""
        seen = set()
        for t_idx, samples in enumerate(self.task_samples):
            task_classes = sorted(list({s['label'] for s in samples}))
            new_classes = [c for c in task_classes if c not in seen]
            revisited_classes = [c for c in task_classes if c in seen]
            sessions = sorted(list({s.get('session', 'unknown') for s in samples}))
            objects = sorted(list({s.get('object_id', 'unknown') for s in samples}))
            
            seen.update(task_classes)

            meta = {
                "task_id": t_idx,
                "num_samples": len(samples),
                "num_classes": len(task_classes),
                "classes": task_classes,
                "new_classes": new_classes,
                "revisited_classes": revisited_classes,
                "cumulative_classes_count": len(seen),
                "sessions": sessions,
                "num_objects": len(objects),
                "objects": objects
            }
            self.task_metadata.append(meta)

    def get_task_metadata(self, task_id: int) -> Dict:
        """Returns metadata summary for a given task."""
        if 0 <= task_id < self.num_tasks:
            return self.task_metadata[task_id]
        raise IndexError(f"Task ID {task_id} out of range [0, {self.num_tasks - 1}]")

    def get_all_tasks_metadata(self) -> List[Dict]:
        """Returns metadata list for all tasks."""
        return self.task_metadata

    def get_train_dataset(self, task_id: int) -> TinyImageNetDataset:
        """Returns TinyImageNetDataset for the training samples of task_id."""
        samples = self.task_samples[task_id]
        return TinyImageNetDataset(
            samples=samples,
            root_dir=self.dm.root_dir,
            transform=get_transforms(is_train=True),
            is_train=True,
            smoke_mode=self.smoke_mode
        )

    def get_train_loader(
        self,
        task_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Returns DataLoader for training task_id."""
        ds = self.get_train_dataset(task_id)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_task_test_dataset(self, task_id: int) -> TinyImageNetDataset:
        """
        Returns test samples for the classes that appeared in task_id.
        Evaluates knowledge retention on the exact concepts learned in task_id.
        """
        target_classes = set(self.task_metadata[task_id]['classes'])
        filtered_test_samples = [
            s for s in self.test_samples if s['label'] in target_classes
        ]
        return TinyImageNetDataset(
            samples=filtered_test_samples,
            root_dir=self.dm.root_dir,
            transform=get_transforms(is_train=False),
            is_train=False,
            smoke_mode=self.smoke_mode
        )

    def get_task_test_loader(
        self,
        task_id: int,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> DataLoader:
        """DataLoader for evaluating performance specifically on task_id test concepts."""
        ds = self.get_task_test_dataset(task_id)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_cumulative_test_dataset(self, up_to_task_id: int) -> TinyImageNetDataset:
        """
        Returns test samples containing all classes observed from task 0 up to up_to_task_id.
        """
        seen = set()
        for t in range(up_to_task_id + 1):
            seen.update(self.task_metadata[t]['classes'])
        filtered = [s for s in self.test_samples if s['label'] in seen]
        return TinyImageNetDataset(
            samples=filtered,
            root_dir=self.dm.root_dir,
            transform=get_transforms(is_train=False),
            is_train=False,
            smoke_mode=self.smoke_mode
        )

    def get_cumulative_test_loader(
        self,
        up_to_task_id: int,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> DataLoader:
        """DataLoader for all concepts seen up to the specified task."""
        ds = self.get_cumulative_test_dataset(up_to_task_id)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_full_test_dataset(self) -> TinyImageNetDataset:
        """Returns the full official test dataset."""
        return TinyImageNetDataset(
            samples=self.test_samples,
            root_dir=self.dm.root_dir,
            transform=get_transforms(is_train=False),
            is_train=False,
            smoke_mode=self.smoke_mode
        )

    def get_full_test_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> DataLoader:
        """DataLoader for the complete official test benchmark."""
        ds = self.get_full_test_dataset()
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_all_train_samples_joint(self) -> List[Dict]:
        """
        Gathers all training samples across all tasks for Joint Upper-Bound training.
        """
        joint_samples = []
        for task_list in self.task_samples:
            joint_samples.extend(task_list)
        return joint_samples

    def get_joint_train_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Returns DataLoader containing all training data simultaneously for Joint baseline."""
        all_samples = self.get_all_train_samples_joint()
        ds = TinyImageNetDataset(
            samples=all_samples,
            root_dir=self.dm.root_dir,
            transform=get_transforms(is_train=True),
            is_train=True,
            smoke_mode=self.smoke_mode
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
