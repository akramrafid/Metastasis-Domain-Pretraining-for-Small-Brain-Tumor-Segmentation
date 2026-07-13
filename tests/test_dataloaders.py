"""
Unit tests for Dataset and DataLoader outputs.
"""
import os
import tempfile
import json
import numpy as np
import nibabel as nib
from src.data.datasets import BrainModalityDataset

def test_dataloader_output_schema() -> None:
    """Verifies that the dataset loads preprocessed cases, handles missing channels via zero-filling, and yields the correct dictionary schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock preprocessed folder structure
        dataset_name = "mock_dataset"
        dataset_dir = os.path.join(tmpdir, dataset_name)
        case_id = "mock_case"
        case_dir = os.path.join(dataset_dir, case_id)
        os.makedirs(case_dir)
        
        # Save a fake 3D image shape
        shape = (8, 8, 8)
        data = np.random.rand(*shape).astype(np.float32)
        # Ensure it has non-zero values
        data[data == 0.0] = 0.1
        affine = np.eye(4)
        
        # Modalities: t1_pre, t1_post, flair are present. t2 is missing!
        nib.save(nib.Nifti1Image(data, affine), os.path.join(case_dir, "t1_pre.nii.gz"))
        nib.save(nib.Nifti1Image(data, affine), os.path.join(case_dir, "t1_post.nii.gz"))
        nib.save(nib.Nifti1Image(data, affine), os.path.join(case_dir, "flair.nii.gz"))
        nib.save(nib.Nifti1Image((data > 0.5).astype(np.uint8), affine), os.path.join(case_dir, "seg.nii.gz"))
        
        # metadata
        with open(os.path.join(case_dir, "metadata.json"), "w") as f:
            json.dump({
                "case_id": case_id,
                "channel_mask": [1, 1, 0, 1]  # Missing T2 channel
            }, f)
            
        # Instantiate dataset
        dataset = BrainModalityDataset(
            dataset_name=dataset_name,
            preprocessed_dir=tmpdir,
            transforms=None
        )
        
        assert len(dataset) == 1
        
        # Fetch the item
        sample = dataset[0]
        assert sample["case_id"] == case_id
        assert sample["source"] == dataset_name
        assert np.array_equal(sample["channel_mask"], [1, 1, 0, 1])
        
        # Check spatial dimensions [4, 8, 8, 8]
        assert sample["image"].shape == (4, 8, 8, 8)
        assert sample["label"].shape == (1, 8, 8, 8)
        
        # Verify that channel index 2 (T2) is completely zero-filled
        assert np.all(sample["image"][2] == 0.0)
        
        # Verify that channel indices 0, 1, 3 contain non-zero data
        assert np.any(sample["image"][0] != 0.0)
        assert np.any(sample["image"][1] != 0.0)
        assert np.any(sample["image"][3] != 0.0)
