"""
Script to run preprocessing across all 4 datasets.
Supports limiting the case count for testing and dry-runs.
"""
import os
import glob
import json
import click
from typing import Dict, Any, List
from src.data.preprocessing import preprocess_case

DATASETS_INFO = {
    "brats_gli": {
        "root": [
            "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
            "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        ],
        "pattern": "*",
        "channels": {
            "t1_pre": "*-t1n.nii.gz",
            "t1_post": "*-t1c.nii.gz",
            "t2": "*-t2w.nii.gz",
            "flair": "*-t2f.nii.gz"
        },
        "channel_mask": [1, 1, 1, 1],
        "label": "*-seg.nii.gz"
    },
    "pretreat_m2b": {
        "root": "PKG - Pretreat-MetsToBrain-Masks/Pretreat-MetsToBrain-Masks",
        "pattern": "*",
        "channels": {
            "t1_pre": "*-t1n.nii.gz",
            "t1_post": "*-t1c.nii.gz",
            "t2": "*-t2w.nii.gz",
            "flair": "*-t2f.nii.gz"
        },
        "channel_mask": [1, 1, 1, 1],
        "label": "*-seg.nii.gz"
    },
    "brainmetshare": {
        "root": "brainmetshare_data/train_NifTI",
        "pattern": "Mets_*",
        "channels": {
            "t1_pre": "t1_pre.nii.gz",
            "t1_post": "t1_gd.nii.gz",
            "t2": None,
            "flair": "flair.nii.gz"
        },
        "channel_mask": [1, 1, 0, 1],
        "label": "seg.nii.gz"
    },
    "brainmetshare_test": {
        "root": "brainmetshare_data/test_NifTI",
        "pattern": "Mets_*",
        "channels": {
            "t1_pre": "t1_pre.nii.gz",
            "t1_post": "t1_gd.nii.gz",
            "t2": None,
            "flair": "flair.nii.gz"
        },
        "channel_mask": [1, 1, 0, 1],
        "label": None
    },
    "ucsf_bmsr": {
        "root": "UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN",
        "pattern": "*",
        "channels": {
            "t1_pre": "_T1pre.nii.gz",
            "t1_post": "_T1post.nii.gz",
            "t2": "_T2Synth.nii.gz",
            "flair": "_FLAIR.nii.gz"
        },
        "channel_mask": [1, 1, 1, 1],
        "label": "_seg.nii.gz"
    }
}

def resolve_channel_paths(sub_dir: str, name: str, sub_name: str, channels: Dict[str, Any], label_pattern: str) -> tuple:
    """Finds exact absolute paths for channels and labels within a subject folder."""
    resolved_channels = {}
    for key, suffix in channels.items():
        if suffix is None:
            resolved_channels[key] = None
            continue
            
        # Format filename pattern
        if name.startswith("brainmetshare"):
            filename = suffix
        else: 
            filename = f"{sub_name}{suffix}"
            
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            resolved_channels[key] = path
        else:
            # Fallback glob check
            matches = glob.glob(os.path.join(sub_dir, f"*{suffix}"))
            resolved_channels[key] = matches[0] if matches else None
            
    # Resolve label
    if label_pattern:
        if name.startswith("brainmetshare"):
            filename = label_pattern
        else:
            filename = f"{sub_name}{label_pattern}"
            
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            label_path = path
        else:
            matches = glob.glob(os.path.join(sub_dir, f"*{label_pattern}"))
            label_path = matches[0] if matches else None
    else:
        label_path = None
        
    return resolved_channels, label_path

@click.command()
@click.option("--limit", default=-1, type=int, help="Limit the number of subjects processed per dataset (-1 for all).")
@click.option("--skip-missing", is_flag=True, help="Skip datasets that are missing on disk instead of raising error.")
def main(limit: int, skip_missing: bool) -> None:
    """Preprocesses all 4 datasets."""
    output_base_dir = "data/preprocessed"
    os.makedirs(output_base_dir, exist_ok=True)
    
    for name, info in DATASETS_INFO.items():
        roots = info["root"]
        if isinstance(roots, str):
            roots = [roots]
            
        subjects = []
        for root in roots:
            if not os.path.exists(root):
                if skip_missing:
                    click.echo(f"Skipping missing directory {root} for dataset {name}...")
                    continue
                else:
                    raise FileNotFoundError(f"Required directory {root} not found.")
            root_subjects = glob.glob(os.path.join(root, info["pattern"]))
            root_subjects = [s for s in root_subjects if os.path.isdir(s)]
            subjects.extend(root_subjects)
            
        subjects.sort()
        
        if limit > 0:
            subjects = subjects[:limit]
            
        click.echo(f"Preprocessing dataset '{name}' ({len(subjects)} cases)...")
        
        for idx, sub in enumerate(subjects):
            sub_name = os.path.basename(sub)
            resolved_channels, label_path = resolve_channel_paths(
                sub, name, sub_name, info["channels"], info["label"]
            )
            
            case_out_dir = os.path.join(output_base_dir, name, sub_name)
            
            try:
                preprocess_case(
                    case_id=sub_name,
                    input_channels=resolved_channels,
                    label_path=label_path,
                    output_dir=case_out_dir,
                    channel_mask=info["channel_mask"]
                )
                if (idx + 1) % 10 == 0 or (idx + 1) == len(subjects):
                    click.echo(f"  Processed {idx + 1}/{len(subjects)} cases...")
            except Exception as e:
                click.echo(f"  Error processing case {sub_name}: {str(e)}", err=True)

    click.echo("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
