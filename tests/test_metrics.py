"""
Unit tests for evaluation metrics.
"""
import torch
from src.evaluation.metrics import (
    compute_dice,
    compute_sensitivity,
    compute_specificity,
    compute_lesion_f1
)

def test_dice_score() -> None:
    """Verifies that Dice metric is computed correctly on synthetic volumes."""
    pred = torch.tensor([[[0, 1], [0, 1]], [[0, 0], [1, 1]]])
    target = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    # Intersection: (0,0,1), (1,1,0), (1,1,1) -> 3 voxels
    # Pred sum: 4, Target sum: 4
    # Dice = 2 * 3 / (4 + 4) = 0.75
    dice = compute_dice(pred, target)
    assert abs(dice - 0.75) < 1e-5

def test_sensitivity_specificity() -> None:
    """Verifies sensitivity and specificity calculations match hand-computed values."""
    pred = torch.tensor([[[0, 1], [0, 1]], [[0, 0], [1, 1]]])
    target = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [1, 1]]])
    # TP: 3, FN: 1 (at (0,1,0)) -> Sens = 3/4 = 0.75
    # TN: 3 (at (0,0,0), (1,0,0), (1,0,1)), FP: 1 (at (0,1,1)) -> Spec = 3/4 = 0.75
    sens = compute_sensitivity(pred, target)
    spec = compute_specificity(pred, target)
    assert abs(sens - 0.75) < 1e-5
    assert abs(spec - 0.75) < 1e-5

def test_lesion_f1() -> None:
    """Verifies lesion-wise F1 calculation matches expected counts."""
    # 2 disjoint lesions in target, 2 in pred
    pred = torch.zeros((10, 10, 10))
    target = torch.zeros((10, 10, 10))
    
    # Target lesion 1
    target[2:4, 2:4, 2:4] = 1
    # Target lesion 2
    target[6:8, 6:8, 6:8] = 1
    
    # Pred lesion 1 (overlaps with target 1)
    pred[3:5, 3:5, 3:5] = 1
    # Pred lesion 2 (no overlap)
    pred[8:9, 8:9, 8:9] = 1
    
    # num_target = 2, num_pred = 2
    # detected_targets = 1 (target 1)
    # detected_predictions = 1 (pred 1)
    # Recall = 1/2 = 0.5, Precision = 1/2 = 0.5 -> F1 = 0.5
    f1 = compute_lesion_f1(pred, target)
    assert abs(f1 - 0.5) < 1e-5
