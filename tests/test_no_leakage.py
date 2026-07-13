"""
Unit test auditing data leakage between self-supervised pretraining and supervised splits.
"""
import os
import json
from src.data.datasets import BrainModalityDataset
from src.training.finetune import get_deterministic_splits

def test_no_leakage_audit() -> None:
    """Audits data leakage in Arm 2's pretraining and fine-tuning splits."""
    preprocessed_dir = "data/preprocessed/pretreat_m2b"
    splits_path = "checkpoints/splits.json"
    
    assert os.path.exists(preprocessed_dir), f"Preprocessed dataset directory {preprocessed_dir} does not exist."
    
    # 1. Load or dynamically compute splits
    if os.path.exists(splits_path):
        with open(splits_path, "r") as f:
            splits = json.load(f)
        train_cases = splits.get("train", [])
        val_cases = splits.get("val", [])
        test_cases = splits.get("test", [])
    else:
        all_cases = sorted(os.listdir(preprocessed_dir))
        train_cases, val_cases, test_cases = get_deterministic_splits(all_cases, seed=42)
        
    val_set = set(val_cases)
    test_set = set(test_cases)
    
    # 2. Instantiate pretraining dataset using the fix (exclude val and test cases)
    pretrain_dataset = BrainModalityDataset(
        dataset_name="pretreat_m2b",
        preprocessed_dir="data/preprocessed",
        exclude_cases=val_cases + test_cases
    )
    pretrain_cases = set(pretrain_dataset.cases)
    
    # 3. Check for overlap
    val_leakage = pretrain_cases.intersection(val_set)
    test_leakage = pretrain_cases.intersection(test_set)
    
    print(f"\n[LEAKAGE AUDIT LOG] Number of corrected pretraining cases: {len(pretrain_cases)}")
    print(f"[LEAKAGE AUDIT LOG] Pretrain vs Val overlap: {len(val_leakage)}/{len(val_set)} cases")
    print(f"[LEAKAGE AUDIT LOG] Pretrain vs Test overlap: {len(test_leakage)}/{len(test_set)} cases")
    
    assert len(val_leakage) == 0, f"Leakage found! Validation cases overlap: {val_leakage}"
    assert len(test_leakage) == 0, f"Leakage found! Test cases overlap: {test_leakage}"
