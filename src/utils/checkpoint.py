"""
Checkpoint saving and loading utilities.
"""
import os
from typing import Dict, Any
import torch
import torch.nn as nn

def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Saves checkpoint state.
    
    Args:
        state: State dictionary to save.
        filepath: Path to save the checkpoint file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_checkpoint(filepath: str, model: nn.Module) -> None:
    """Loads checkpoint state into the model.
    
    Args:
        filepath: Path to the checkpoint file.
        model: PyTorch model module to load state into.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    checkpoint = torch.load(filepath, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
