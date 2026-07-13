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
    # Globally patch MONAI transform seed limits for Windows 32-bit signed C-long compatibility
    try:
        import monai
        import monai.utils
        import monai.utils.misc
        import monai.transforms.compose
        import monai.transforms.transform
        
        win_max_seed = 2147483647
        monai.utils.MAX_SEED = win_max_seed
        monai.utils.misc.MAX_SEED = win_max_seed
        monai.transforms.compose.MAX_SEED = win_max_seed
        monai.transforms.transform.MAX_SEED = win_max_seed
        
        monai.utils.set_determinism(seed=seed)
    except ImportError:
        pass
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
