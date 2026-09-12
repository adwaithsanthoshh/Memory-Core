"""
Replay Model and Rehearsal integration for Continual Learning.
Combines current task data streams with buffered historical exemplars.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from continual.memory import ReplayMemoryBuffer


def compute_replay_loss(
    model: nn.Module,
    memory_buffer: ReplayMemoryBuffer,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module
) -> Optional[torch.Tensor]:
    """
    Samples from memory buffer and calculates cross-entropy rehearsal loss.
    """
    batch = memory_buffer.sample_batch(batch_size=batch_size)
    if batch is None:
        return None

    mem_imgs, mem_targets, _ = batch
    mem_imgs = mem_imgs.to(device)
    mem_targets = mem_targets.to(device)

    mem_outputs = model(mem_imgs)
    replay_loss = criterion(mem_outputs, mem_targets)
    return replay_loss
