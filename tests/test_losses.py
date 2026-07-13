"""
Unit tests for custom loss functions.
"""
import torch
from src.training.losses import SensitivitySpecificityLoss

def test_sens_spec_tradeoff_loss() -> None:
    """Verifies that the sensitivity-specificity tradeoff loss behaves as expected."""
    loss_fn = SensitivitySpecificityLoss(r=0.05, include_background=True)
    # Shape: [B, C, H, W, D]
    pred = torch.zeros((1, 2, 4, 4, 4), requires_grad=True)
    target = torch.zeros((1, 2, 4, 4, 4))
    
    # Put a target voxel
    target[0, 1, 2, 2, 2] = 1.0
    
    # Calculate loss
    loss = loss_fn(pred, target)
    assert loss.item() > 0.0
    
    # Verify backward pass works
    loss.backward()
    assert pred.grad is not None
