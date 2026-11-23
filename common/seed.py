"""Seed the random number generators used by the course experiments."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch and request deterministic cuDNN convolutions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
