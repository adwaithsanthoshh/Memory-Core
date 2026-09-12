"""
Memory Buffer for Experience Replay.
Supports fixed memory budgets: [50, 100, 250, 500, 1000].
Implements reservoir sampling and class-balanced rehearsal strategies.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple


class ReplayMemoryBuffer:
    """
    Fixed-capacity memory buffer for continual rehearsal.
    
    Guarantees:
    - Maximum capacity is strictly enforced (budget <= capacity).
    - Random/Reservoir replacement strategy.
    - Retrieval of memory samples as DataLoader or batch tensor.
    """
    def __init__(self, capacity: int = 250):
        self.capacity = capacity
        self.storage: List[Tuple[torch.Tensor, int, Dict]] = []
        self.total_seen_samples = 0

    def __len__(self):
        return len(self.storage)

    def add_sample(self, img_tensor: torch.Tensor, label: int, meta: Dict):
        """
        Adds sample using reservoir sampling to maintain uniform representation over time.
        """
        self.total_seen_samples += 1
        item = (img_tensor.detach().cpu(), label, meta)

        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            # Reservoir replacement with probability capacity / total_seen
            idx = random.randint(0, self.total_seen_samples - 1)
            if idx < self.capacity:
                self.storage[idx] = item

    def add_from_task(self, dataset: Dataset, num_to_add: Optional[int] = None):
        """
        Populates buffer from a task dataset.
        """
        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)

        limit = num_to_add if num_to_add is not None else n
        for i in indices[:limit]:
            img, label, meta = dataset[i]
            self.add_sample(img, label, meta)

    def sample_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[Dict]]]:
        """
        Samples a mini-batch of stored memories.
        """
        if len(self.storage) == 0:
            return None

        actual_k = min(batch_size, len(self.storage))
        sampled = random.sample(self.storage, actual_k)
        
        imgs = torch.stack([x[0] for x in sampled])
        labels = torch.tensor([x[1] for x in sampled], dtype=torch.long)
        metas = [x[2] for x in sampled]
        
        return imgs, labels, metas

    def get_all_samples(self) -> List[Tuple[torch.Tensor, int, Dict]]:
        """Returns all stored samples for inspection and visualization."""
        return self.storage
