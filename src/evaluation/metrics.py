"""
Segmentation evaluation metrics (Dice, Sensitivity, Specificity, HD95, and lesion-wise F1).
"""
import torch
import numpy as np
from scipy.ndimage import label
from monai.metrics import compute_hausdorff_distance

def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Calculates Dice coefficient on binary tensors.
    
    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Dice coefficient value.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    intersection = torch.sum(pred_flat * target_flat)
    return float((2.0 * intersection + eps) / (torch.sum(pred_flat) + torch.sum(target_flat) + eps))

def compute_sensitivity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Calculates Sensitivity (True Positive Rate).
    
    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Sensitivity value.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    tp = torch.sum(pred_flat * target_flat)
    fn = torch.sum((1.0 - pred_flat) * target_flat)
    return float((tp + eps) / (tp + fn + eps))

def compute_specificity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Calculates Specificity (True Negative Rate).
    
    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        eps: Small epsilon for numerical stability.
        
    Returns:
        Specificity value.
    """
    pred_flat = pred.contiguous().view(-1).float()
    target_flat = target.contiguous().view(-1).float()
    tn = torch.sum((1.0 - pred_flat) * (1.0 - target_flat))
    fp = torch.sum(pred_flat * (1.0 - target_flat))
    return float((tn + eps) / (tn + fp + eps))

def compute_hd95(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates 95th percentile Hausdorff Distance using MONAI.
    
    Args:
        pred: Binary prediction tensor of shape [H, W, D].
        target: Binary target tensor of shape [H, W, D].
        
    Returns:
        HD95 value in mm (assumes isotropic spacing).
    """
    # MONAI expects inputs of shape [B, C, H, W, D] or [B, C, spatial_shape...]
    p_unsqueezed = pred.unsqueeze(0).unsqueeze(0)
    t_unsqueezed = target.unsqueeze(0).unsqueeze(0)
    try:
        hd = compute_hausdorff_distance(p_unsqueezed, t_unsqueezed, percentile=95.0)
        return float(hd.item())
    except Exception:
        # Return nan or infinity if calculation fails (e.g. empty prediction or target)
        return float('nan')

def compute_lesion_f1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates lesion-wise F1 score based on connected components.
    
    A predicted lesion is a TP if it overlaps by >= 1 voxel with a target lesion.
    
    Args:
        pred: Binary prediction tensor.
        target: Binary target tensor.
        
    Returns:
        Lesion-wise F1 score.
    """
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    target_np = target.detach().cpu().numpy().astype(np.uint8)
    
    # Label connected components
    pred_labels, num_pred = label(pred_np)
    target_labels, num_target = label(target_np)
    
    if num_target == 0 and num_pred == 0:
        return 1.0
    if num_target == 0 or num_pred == 0:
        return 0.0
        
    # Check overlap of predicted components with target components
    detected_targets = 0
    for t_idx in range(1, num_target + 1):
        target_mask = (target_labels == t_idx)
        # If prediction overlaps with this target lesion
        if np.sum(pred_np[target_mask]) > 0:
            detected_targets += 1
            
    detected_predictions = 0
    for p_idx in range(1, num_pred + 1):
        pred_mask = (pred_labels == p_idx)
        # If target overlaps with this predicted lesion
        if np.sum(target_np[pred_mask]) > 0:
            detected_predictions += 1
            
    recall = detected_targets / num_target
    precision = detected_predictions / num_pred
    
    if precision + recall == 0:
        return 0.0
        
    return 2.0 * precision * recall / (precision + recall)
