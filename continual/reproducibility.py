"""
Reproducibility utilities for MEMORYCORE.
Enforces seed=42 across Python, NumPy, PyTorch CPU/CUDA, and cuDNN.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Sets random seeds for end-to-end reproducibility.
    
    Args:
        seed: Random seed (Hackathon mandatory default: 42).
        deterministic: Enables deterministic algorithms in PyTorch / cuDNN.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic and hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolves execution device.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
