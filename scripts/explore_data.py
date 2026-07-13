"""
Data exploration and validation script.
Walks each dataset folder, checks file counts, reads NIfTI headers,
and prints a structured summary table.
"""
import os
import glob
import random
from typing import Dict, List, Tuple, Any
import nibabel as nib
import pandas as pd

# Define expected dataset locations (relative to workspace root)
DATASETS = {
    "BraTS-GLI-Train": {
        "path": "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
        "pattern": "*",
        "is_dir": True,
        "files": ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]
    },
    "Pretreat-MetsToBrain": {
        "path": "PKG - Pretreat-MetsToBrain-Masks/Pretreat-MetsToBrain-Masks",
        "pattern": "*",
        "is_dir": True,
        "files": ["-t1n.nii.gz", "-t1c.nii.gz", "-t2w.nii.gz", "-t2f.nii.gz", "-seg.nii.gz"]
    },
    "BrainMetShare-Train": {
        "path": "brainmetshare_data/train_NifTI",
        "pattern": "Mets_*",
        "is_dir": True,
        "files": ["t1_pre.nii.gz", "t1_gd.nii.gz", "flair.nii.gz", "seg.nii.gz"]
    },
    "UCSF-BMSR-Train": {
        "path": "UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN",
        "pattern": "*",
        "is_dir": True,
        "files": ["_T1pre.nii.gz", "_T1post.nii.gz", "_T2Synth.nii.gz", "_FLAIR.nii.gz", "_seg.nii.gz"]
    }
}

def check_nifti_header(filepath: str) -> Tuple[Tuple[int, ...], Tuple[float, ...], str]:
    """Reads shape, spacing, and orientation from a NIfTI header."""
    try:
        img = nib.load(filepath)
        header = img.header
        shape = img.shape
        spacing = tuple(float(x) for x in header.get_zooms())
        # Basic orientation check
        affine = img.affine
        orientation = "".join(nib.aff2axcodes(affine))
        return shape, spacing, orientation
    except Exception as e:
        raise ValueError(f"Error reading header of {filepath}: {str(e)}")

def validate_dataset(name: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a single dataset and returns summary statistics."""
    root = info["path"]
    if not os.path.exists(root):
        return {"status": "Missing", "error": f"Path {root} does not exist"}

    subjects = glob.glob(os.path.join(root, info["pattern"]))
    subjects = [s for s in subjects if os.path.isdir(s)]
    total_subjects = len(subjects)

    if total_subjects == 0:
        return {"status": "Empty", "error": f"No subjects matched pattern {info['pattern']}"}

    missing_files_count = 0
    inconsistent_shapes = set()
    inconsistent_spacings = set()
    inconsistent_orientations = set()
    corrupted_files = []

    # Sample up to 10 subjects for detailed header check to save memory/time
    sampled_subjects = random.sample(subjects, min(10, total_subjects))
    sample_details = []

    for sub in subjects:
        sub_name = os.path.basename(sub)
        # Check files existence
        for suffix in info["files"]:
            # For UCSF, suffix contains subject name prefix sometimes, so we check endswith or match
            if name in ["BraTS-GLI-Train", "Pretreat-MetsToBrain"]:
                expected_filename = f"{sub_name}{suffix}"
            elif name == "UCSF-BMSR-Train":
                expected_filename = f"{sub_name}{suffix}"
            else: # BrainMetShare-Train
                expected_filename = suffix

            filepath = os.path.join(sub, expected_filename)
            if not os.path.exists(filepath):
                # Try finding if there's any close match
                matches = glob.glob(os.path.join(sub, f"*{suffix}"))
                if matches:
                    filepath = matches[0]
                else:
                    missing_files_count += 1
                    continue

            # If sampled, check NIfTI header
            if sub in sampled_subjects:
                try:
                    shape, spacing, orient = check_nifti_header(filepath)
                    sample_details.append({
                        "subject": sub_name,
                        "file": os.path.basename(filepath),
                        "shape": shape,
                        "spacing": spacing,
                        "orient": orient
                    })
                    # Exclude labels/seg masks from modality shapes consistency comparison if they differ
                    if "seg" not in suffix:
                        inconsistent_shapes.add(shape)
                        inconsistent_spacings.add(spacing)
                        inconsistent_orientations.add(orient)
                except Exception as e:
                    corrupted_files.append((filepath, str(e)))

    return {
        "status": "Success" if (missing_files_count == 0 and len(corrupted_files) == 0) else "Incomplete/Issues",
        "total_subjects": total_subjects,
        "missing_files": missing_files_count,
        "corrupted_files": len(corrupted_files),
        "shapes": list(inconsistent_shapes),
        "spacings": list(inconsistent_spacings),
        "orientations": list(inconsistent_orientations),
        "sample_details": sample_details
    }

def main():
    print("==================================================")
    print("Starting Dataset Inventory & Validation (Phase 1)")
    print("==================================================")
    
    results = {}
    for name, info in DATASETS.items():
        print(f"Validating {name}...")
        results[name] = validate_dataset(name, info)

    # Output Markdown summary table
    print("\n## DATASET VALIDATION SUMMARY TABLE\n")
    print("| Dataset | Status | Case Count | Missing Files | Corrupted Files | Sample Shapes | Sample Spacings (mm) | Orientations |")
    print("|---|---|---|---|---|---|---|---|")
    
    for name, res in results.items():
        if res.get("status") == "Missing":
            print(f"| {name} | **Missing** | 0 | - | - | - | - | - |")
        elif res.get("status") == "Empty":
            print(f"| {name} | **Empty** | 0 | - | - | - | - | - |")
        else:
            shapes_str = ", ".join([str(s) for s in res["shapes"][:2]])
            spacings_str = ", ".join([f"({', '.join([f'{x:.2f}' for x in sp])})" for sp in res["spacings"][:2]])
            orients_str = ", ".join(res["orientations"])
            print(f"| {name} | {res['status']} | {res['total_subjects']} | {res['missing_files']} | {res['corrupted_files']} | {shapes_str} | {spacings_str} | {orients_str} |")

    # Log specific errors or warnings
    has_issues = False
    for name, res in results.items():
        if res.get("status") != "Success" and res.get("status") != "Missing":
            has_issues = True
            print(f"\n### Issues found in {name}:")
            if res.get("missing_files", 0) > 0:
                print(f"- Missing {res['missing_files']} expected NIfTI file(s).")
            if res.get("corrupted_files", 0) > 0:
                print(f"- Found {res['corrupted_files']} corrupted NIfTI files.")
    
    if not has_issues:
        print("\nAll datasets validated successfully! Ready to proceed to preprocessing.")

if __name__ == "__main__":
    main()
