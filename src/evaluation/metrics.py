"""
Segmentation evaluation metrics (Dice, Sensitivity, Specificity, HD95, and lesion-wise F1).
"""
import torch
import numpy as np
from scipy.ndimage import label
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

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

def compute_lesion_wise_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: np.ndarray
) -> dict:
    """Computes lesion-wise evaluation metrics matching the BraTS Challenge Task 4 convention.
    
    Checks that the spacing is 1.0mm isotropic, labels connected components using 26-connectivity,
    filters components based on a volume size threshold of 27 voxels (27 mm^3), and computes:
      - lesion_wise_dice (on large components >= 27 voxels)
      - lesion_wise_nsd (Normalized Surface Distance with 1.0mm tolerance on large components)
      - lesion_wise_f1 (F1 score on large components)
      - small_lesion_recall (Recall on small target components < 27 voxels)
      - false_positive_lesions_count (Number of predicted components of any size with zero GT overlap)
      
    Args:
      - pred: Binary prediction tensor of shape [H, W, D].
      - target: Binary target tensor of shape [H, W, D].
      - spacing: Spacing of the volume [x, y, z] in mm.
        
    Returns:
        Dictionary containing the computed lesion-wise metrics.
    """
    # Explicitly assert that the spacing is 1.0mm isotropic
    assert np.allclose(spacing, 1.0, atol=1e-3), f"Spacing is not 1.0mm isotropic: {spacing}"
    
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    target_np = target.detach().cpu().numpy().astype(np.uint8)
    
    # 26-connectivity neighborhood structure for 3D
    structure_26 = np.ones((3, 3, 3), dtype=np.uint8)
    
    pred_labels, num_pred = label(pred_np, structure=structure_26)
    target_labels, num_target = label(target_np, structure=structure_26)
    
    # 1. Size counting via np.bincount (extremely fast in C)
    target_sizes = np.bincount(target_labels.ravel())
    large_target_indices = [i for i in range(1, num_target + 1) if target_sizes[i] >= 27]
    small_target_indices = [i for i in range(1, num_target + 1) if target_sizes[i] < 27]
    
    pred_sizes = np.bincount(pred_labels.ravel())
    large_pred_indices = [i for i in range(1, num_pred + 1) if pred_sizes[i] >= 27]
    
    # 2. Create large-component binary masks via np.isin (vectorized)
    target_large_mask = np.isin(target_labels, large_target_indices).astype(np.uint8)
    pred_large_mask = np.isin(pred_labels, large_pred_indices).astype(np.uint8)
    
    pred_large_tensor = torch.from_numpy(pred_large_mask)
    target_large_tensor = torch.from_numpy(target_large_mask)
    
    # 1. Lesion-wise Dice (computed on large masks)
    lesion_wise_dice = compute_dice(pred_large_tensor, target_large_tensor)
    
    # 2. Lesion-wise NSD (computed on large masks using MONAI compute_surface_dice)
    num_large_target_vox = int(np.sum(target_large_mask))
    num_large_pred_vox = int(np.sum(pred_large_mask))
    
    if num_large_target_vox == 0 and num_large_pred_vox == 0:
        lesion_wise_nsd = 1.0
    elif num_large_target_vox == 0 or num_large_pred_vox == 0:
        lesion_wise_nsd = 0.0
    else:
        # MONAI expects shape [B, C, H, W, D]
        p_unsqueezed = pred_large_tensor.unsqueeze(0).unsqueeze(0).float()
        t_unsqueezed = target_large_tensor.unsqueeze(0).unsqueeze(0).float()
        try:
            nsd_val = compute_surface_dice(p_unsqueezed, t_unsqueezed, class_thresholds=[1.0], spacing=1.0)
            lesion_wise_nsd = float(nsd_val.item())
        except Exception:
            lesion_wise_nsd = 0.0
            
    # 3. Lesion-wise F1 (computed on large components)
    num_large_target = len(large_target_indices)
    num_large_pred = len(large_pred_indices)
    
    if num_large_target == 0 and num_large_pred == 0:
        lesion_wise_f1 = 1.0
    elif num_large_target == 0 or num_large_pred == 0:
        lesion_wise_f1 = 0.0
    else:
        from scipy.ndimage import sum as nd_sum
        # Overlaps computed using nd_sum (vectorized)
        overlap_sums = nd_sum(pred_large_mask, target_labels, large_target_indices)
        if isinstance(overlap_sums, (float, int, np.integer, np.floating)):
            overlap_sums = np.array([overlap_sums])
        else:
            overlap_sums = np.asarray(overlap_sums)
        tp = int(np.sum(overlap_sums > 0))
        
        overlap_sums_pred = nd_sum(target_large_mask, pred_labels, large_pred_indices)
        if isinstance(overlap_sums_pred, (float, int, np.integer, np.floating)):
            overlap_sums_pred = np.array([overlap_sums_pred])
        else:
            overlap_sums_pred = np.asarray(overlap_sums_pred)
        fp = int(np.sum(overlap_sums_pred == 0))
        
        fn = num_large_target - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            lesion_wise_f1 = 2.0 * precision * recall / (precision + recall)
        else:
            lesion_wise_f1 = 0.0
            
    # 4. Small Lesion Recall (< 27 voxels)
    if len(small_target_indices) == 0:
        small_lesion_recall = float('nan')
    else:
        from scipy.ndimage import sum as nd_sum
        small_overlap_sums = nd_sum(pred_np, target_labels, small_target_indices)
        if isinstance(small_overlap_sums, (float, int, np.integer, np.floating)):
            small_overlap_sums = np.array([small_overlap_sums])
        else:
            small_overlap_sums = np.asarray(small_overlap_sums)
        small_tp = int(np.sum(small_overlap_sums > 0))
        small_lesion_recall = small_tp / len(small_target_indices)
        
    # 5. False Positive Lesion Count (number of predicted components >= 27 voxels with zero target overlap)
    if len(large_pred_indices) > 0:
        from scipy.ndimage import sum as nd_sum
        fp_sums = nd_sum(target_np, pred_labels, large_pred_indices)
        if isinstance(fp_sums, (float, int, np.integer, np.floating)):
            fp_sums = np.array([fp_sums])
        else:
            fp_sums = np.asarray(fp_sums)
        false_positive_lesions_count = int(np.sum(fp_sums == 0))
    else:
        false_positive_lesions_count = 0
        
    # 6. False Positive Speckle Count (number of predicted components of any size with zero target overlap)
    if num_pred > 0:
        from scipy.ndimage import sum as nd_sum
        pred_indices = list(range(1, num_pred + 1))
        all_fp_sums = nd_sum(target_np, pred_labels, pred_indices)
        if isinstance(all_fp_sums, (float, int, np.integer, np.floating)):
            all_fp_sums = np.array([all_fp_sums])
        else:
            all_fp_sums = np.asarray(all_fp_sums)
        false_positive_speckle_count = int(np.sum(all_fp_sums == 0))
    else:
        false_positive_speckle_count = 0
            
    return {
        "lesion_wise_dice": lesion_wise_dice,
        "lesion_wise_nsd": lesion_wise_nsd,
        "lesion_wise_f1": lesion_wise_f1,
        "small_lesion_recall": small_lesion_recall,
        "false_positive_lesions_count": float(false_positive_lesions_count),
        "false_positive_speckle_count": float(false_positive_speckle_count)
    }
