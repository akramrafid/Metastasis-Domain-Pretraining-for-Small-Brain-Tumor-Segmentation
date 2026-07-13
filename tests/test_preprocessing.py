"""
Unit tests for preprocessing pipeline correctness.
"""
import os
import tempfile
import numpy as np
import nibabel as nib
from src.data.preprocessing import preprocess_case

def test_preprocessing_isotropic_resample() -> None:
    """Verifies that preprocessing resamples correctly to isotropic 1mm^3 resolution and RAS orientation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock raw subject folder
        sub_dir = os.path.join(tmpdir, "raw_sub")
        os.makedirs(sub_dir)
        
        # Write fake NIfTI images (with non-isotropic spacing, e.g. 0.9 x 0.9 x 1.2)
        # Spatial size is 8x8x8
        data = np.random.rand(8, 8, 8).astype(np.float32)
        # Add a zero background border to mock brain extraction region
        data[0, :, :] = 0.0
        data[-1, :, :] = 0.0
        
        affine = np.diag([0.9, 0.9, 1.2, 1.0])
        # Set orientation to LPS (by putting negative signs on x, y elements)
        affine[0, 0] = -0.9
        affine[1, 1] = -0.9
        
        input_channels = {}
        for key in ["t1_pre", "t1_post", "t2", "flair"]:
            path = os.path.join(sub_dir, f"{key}.nii.gz")
            nib.save(nib.Nifti1Image(data, affine), path)
            input_channels[key] = path
            
        label_path = os.path.join(sub_dir, "seg.nii.gz")
        label_data = (data > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(label_data, affine), label_path)
        
        out_dir = os.path.join(tmpdir, "preprocessed_sub")
        
        # Run preprocess_case
        preprocess_case(
            case_id="mock_sub",
            input_channels=input_channels,
            label_path=label_path,
            output_dir=out_dir,
            channel_mask=[1, 1, 1, 1]
        )
        
        # Assert all files are present
        for key in ["t1_pre", "t1_post", "t2", "flair", "seg"]:
            path = os.path.join(out_dir, f"{key}.nii.gz")
            assert os.path.exists(path), f"Preprocessed file {path} was not created."
            
            # Check spacing is 1.0mm isotropic
            img = nib.load(path)
            zooms = img.header.get_zooms()
            assert np.allclose(zooms, (1.0, 1.0, 1.0), atol=1e-3), f"Spacing was {zooms}, expected isotropic 1.0."
            
            # Check orientation is RAS
            orientation = "".join(nib.aff2axcodes(img.affine))
            assert orientation == "RAS", f"Orientation was {orientation}, expected RAS."
            
            # Verify Z-score normalization: check that mean is close to 0 and std is close to 1
            if key != "seg":
                img_data = img.get_fdata()
                # Exclude zero background
                non_zero = img_data[img_data != 0]
                if len(non_zero) > 0:
                    assert abs(np.mean(non_zero)) < 1.0, f"Mean not normalized: {np.mean(non_zero)}"
                    assert abs(np.std(non_zero) - 1.0) < 0.2, f"Std not normalized: {np.std(non_zero)}"
