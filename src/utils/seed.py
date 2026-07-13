"""
Seeding function for reproducibility.
"""
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Sets seed for random, numpy, and torch to ensure full determinism.
    
    Args:
        seed: The integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
