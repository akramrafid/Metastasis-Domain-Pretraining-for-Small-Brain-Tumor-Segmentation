"""
Inference and evaluation across cohorts using sliding window inference.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from src.data.datasets import BrainModalityDataset
from src.evaluation.metrics import (
    compute_dice,
    compute_sensitivity,
    compute_specificity,
    compute_hd95,
    compute_lesion_f1
)
from src.models.swin_unetr import SwinUNETRWrapper
from src.utils.logging_utils import setup_logger

logger = setup_logger("evaluator")

def run_evaluation(
    model_path: str,
    dataset_name: str,
    preprocessed_dir: str = "data/preprocessed",
    allowed_cases: Optional[List[str]] = None,
    device: str = "cuda",
    labeled: bool = True
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Runs inference and calculates performance metrics on a dataset.
    
    Args:
        model_path: Path to best fine-tuned model checkpoint.
        dataset_name: Name of preprocessed dataset to evaluate.
        preprocessed_dir: Path to preprocessed data root directory.
        allowed_cases: Optional list of case IDs to evaluate (e.g., test split cases).
        device: Device to run inference on.
        labeled: Whether the target dataset has labels to compute Dice, etc.
        
    Returns:
        Tuple of:
            - summary_metrics: Dict containing mean values for each metric.
            - patient_metrics: Dict mapping case_id to its individual metric dict.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load base dataset
    dataset = BrainModalityDataset(dataset_name=dataset_name, preprocessed_dir=preprocessed_dir)
    
    # Filter cases if allowed_cases is provided
    if allowed_cases is not None:
        allowed_set = set(allowed_cases)
        dataset.cases = [c for c in dataset.cases if c in allowed_set]
        
    if len(dataset) == 0:
        logger.warning(f"No cases found for evaluation on dataset {dataset_name}.")
        return {}, {}
        
    logger.info(f"Evaluating model {model_path} on dataset {dataset_name} ({len(dataset)} cases)...")
    
    # Initialize model wrapper
    dummy_config = {"in_channels": 4, "out_channels": 3, "feature_size": 48, "use_checkpoint": False}
    model = SwinUNETRWrapper(config=dummy_config)
    
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device_obj)
    model.eval()
    
    patient_metrics = {}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            case_id = batch["case_id"]
            
            # Inputs shape [C, H, W, D], add batch dimension [1, C, H, W, D]
            images = torch.from_numpy(batch["image"]).unsqueeze(0).to(device_obj)
            labels = torch.from_numpy(batch["label"]).unsqueeze(0).to(device_obj)
            
            # Sliding window inference
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.25,
                device=device_obj
            )
            
            # Apply argmax or soft thresholding to get binary labels
            pred_labels = torch.argmax(outputs, dim=1)  # [1, H, W, D]
            
            pred_binary = (pred_labels > 0)
            pred_3d = pred_binary[0]
            
            if labeled:
                # Standard multi-class target is usually [1, 1, H, W, D], we squeeze channel dimension
                target_labels = torch.argmax(labels, dim=1) if labels.shape[1] > 1 else labels.squeeze(1) # [1, H, W, D]
                target_binary = (target_labels > 0)
                target_3d = target_binary[0]
                
                # Calculate metrics
                dice = compute_dice(pred_3d, target_3d)
                sens = compute_sensitivity(pred_3d, target_3d)
                spec = compute_specificity(pred_3d, target_3d)
                hd95 = compute_hd95(pred_3d, target_3d)
                lesion_f1 = compute_lesion_f1(pred_3d, target_3d)
                
                patient_metrics[case_id] = {
                    "dice": dice,
                    "sensitivity": sens,
                    "specificity": spec,
                    "hd95": hd95,
                    "lesion_f1": lesion_f1
                }
            else:
                # Unlabeled case: calculate detection rate (1.0 if any predicted lesion, 0.0 otherwise)
                has_detection = float(torch.sum(pred_3d).item() > 0)
                patient_metrics[case_id] = {
                    "detection_rate": has_detection
                }
            
            # Log progress
            if (idx + 1) % 5 == 0 or (idx + 1) == len(dataset):
                logger.info(f"  Evaluated {idx + 1}/{len(dataset)} cases...")
                
    # Calculate summary metrics (means)
    summary_metrics = {}
    if labeled:
        metrics_keys = ["dice", "sensitivity", "specificity", "hd95", "lesion_f1"]
        for k in metrics_keys:
            values = [p[k] for p in patient_metrics.values() if not np.isnan(p[k])]
            if values:
                summary_metrics[f"mean_{k}"] = float(np.mean(values))
                summary_metrics[f"std_{k}"] = float(np.std(values))
            else:
                summary_metrics[f"mean_{k}"] = float('nan')
                summary_metrics[f"std_{k}"] = float('nan')
    else:
        values = [p["detection_rate"] for p in patient_metrics.values()]
        if values:
            summary_metrics["mean_detection_rate"] = float(np.mean(values))
            summary_metrics["std_detection_rate"] = float(np.std(values))
        else:
            summary_metrics["mean_detection_rate"] = float('nan')
            summary_metrics["std_detection_rate"] = float('nan')
            
    return summary_metrics, patient_metrics
