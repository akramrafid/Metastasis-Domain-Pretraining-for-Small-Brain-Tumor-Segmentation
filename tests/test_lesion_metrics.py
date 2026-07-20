"""
Unit tests for the new lesion-wise evaluation metrics.
"""
import torch
import numpy as np
from src.evaluation.metrics import compute_lesion_wise_metrics

def test_spacing_assertion() -> None:
    """Verifies that the spacing assertion correctly fails loudly for non-1.0mm spacing."""
    pred = torch.zeros((5, 5, 5))
    target = torch.zeros((5, 5, 5))
    
    # Non-isotropic spacing should fail
    non_isotropic = np.array([1.0, 1.2, 1.0])
    try:
        compute_lesion_wise_metrics(pred, target, non_isotropic)
        raise AssertionError("Failed to raise AssertionError for non-isotropic spacing.")
    except AssertionError as e:
        assert "Spacing is not 1.0mm isotropic" in str(e)
    
    # 2.0mm isotropic spacing should also fail
    isotropic_2mm = np.array([2.0, 2.0, 2.0])
    try:
        compute_lesion_wise_metrics(pred, target, isotropic_2mm)
        raise AssertionError("Failed to raise AssertionError for 2.0mm spacing.")
    except AssertionError as e:
        assert "Spacing is not 1.0mm isotropic" in str(e)
    
    # Exactly 1.0mm spacing should pass
    exactly_1mm = np.array([1.0, 1.0, 1.0])
    try:
        compute_lesion_wise_metrics(pred, target, exactly_1mm)
    except AssertionError:
        raise AssertionError("AssertionError raised for exactly 1.0mm isotropic spacing.")

def test_connected_component_and_metrics() -> None:
    """Verifies connected component metrics on synthetic 3D volumes.
    
    We setup:
      - 1 large target lesion: size 3x3x3 = 27 voxels (large)
      - 1 small target lesion: size 2x2x2 = 8 voxels (small)
      - 1 large prediction: size 3x3x3 = 27 voxels (large) overlapping large target perfectly
      - 1 small prediction: size 2x2x2 = 8 voxels (small) overlapping small target perfectly
      - 1 false positive prediction: size 2x2x2 = 8 voxels (small) disjoint from target
    """
    pred = torch.zeros((12, 12, 12))
    target = torch.zeros((12, 12, 12))
    
    # 1. Large target lesion at [1:4, 1:4, 1:4] (size 27 >= 27)
    target[1:4, 1:4, 1:4] = 1
    # 2. Small target lesion at [7:9, 7:9, 7:9] (size 8 < 27)
    target[7:9, 7:9, 7:9] = 1
    
    # 1. Large pred overlapping large target (size 27 >= 27)
    pred[1:4, 1:4, 1:4] = 1
    # 2. Small pred overlapping small target (size 8 < 27)
    pred[7:9, 7:9, 7:9] = 1
    # 3. Disjoint small false positive prediction at [9:11, 1:3, 9:11] (size 8 < 27)
    # This is disjoint from target lesions
    pred[9:11, 1:3, 9:11] = 1
    # 4. Disjoint large false positive prediction at [1:4, 7:10, 7:10] (size 27 >= 27)
    # This is also disjoint from target lesions
    pred[1:4, 7:10, 7:10] = 1
    
    spacing = np.array([1.0, 1.0, 1.0])
    metrics = compute_lesion_wise_metrics(pred, target, spacing)
    
    # Large metrics:
    # Target large mask has target[1:4, 1:4, 1:4] (1 lesion, size 27)
    # Pred large mask has pred[1:4, 1:4, 1:4] (size 27) and pred[1:4, 7:10, 7:10] (size 27)
    # Target has 27 voxels, Pred has 54 voxels, intersection is 27 voxels.
    # Lesion-wise voxel Dice on large components = 2 * 27 / (27 + 54) = 54 / 81 = 0.666667
    assert abs(metrics["lesion_wise_dice"] - 0.666667) < 1e-5
    assert metrics["lesion_wise_nsd"] < 1.0
    # Lesion-wise F1: TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0 -> F1=0.66667
    assert abs(metrics["lesion_wise_f1"] - 0.666667) < 1e-5
    
    # Small target recall:
    # There is 1 small target lesion of size 8. It overlaps with pred[7:9, 7:9, 7:9].
    # So Small Lesion Recall = 1.0 (1/1)
    assert abs(metrics["small_lesion_recall"] - 1.0) < 1e-5
    
    # False Positive predicted components:
    # - pred[1:4, 1:4, 1:4] overlaps with GT (not FP)
    # - pred[7:9, 7:9, 7:9] overlaps with GT (not FP)
    # - pred[9:11, 1:3, 9:11] has zero target overlap but is size 8 < 27 (not counted in large FP, but is a speckle FP)
    # - pred[9:12, 4:7, 9:12] has zero target overlap and is size 27 >= 27 (is a large FP!)
    # Total large FP count = 1
    # Total speckle FP count = 2
    assert int(metrics["false_positive_lesions_count"]) == 1
    assert int(metrics["false_positive_speckle_count"]) == 2

def test_small_lesion_recall_undetected() -> None:
    """Verifies that undetected small lesions are correctly reported."""
    pred = torch.zeros((10, 10, 10))
    target = torch.zeros((10, 10, 10))
    
    # 1 small target lesion at [1:3, 1:3, 1:3] (size 8 < 27)
    target[1:3, 1:3, 1:3] = 1
    # 1 small target lesion at [7:9, 7:9, 7:9] (size 8 < 27)
    target[7:9, 7:9, 7:9] = 1
    
    # Pred only overlaps with the first small target
    pred[1:3, 1:3, 1:3] = 1
    
    spacing = np.array([1.0, 1.0, 1.0])
    metrics = compute_lesion_wise_metrics(pred, target, spacing)
    
    # Small lesion recall: 1 detected / 2 total = 0.5
    assert abs(metrics["small_lesion_recall"] - 0.5) < 1e-5
    
    # No large components, so large metrics should return 1.0 or nan/empty defaults
    assert abs(metrics["lesion_wise_dice"] - 1.0) < 1e-5
    assert abs(metrics["lesion_wise_f1"] - 1.0) < 1e-5
    assert abs(metrics["lesion_wise_nsd"] - 1.0) < 1e-5
