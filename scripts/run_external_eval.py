"""
Execution script for running evaluation on metastasis test split, Stanford BrainMetShare, and UCSF-BMSR.
Saves metrics for statistical significance testing.
"""
import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from src.evaluation.evaluator import run_evaluation
from src.utils.logging_utils import setup_logger

logger = setup_logger("run_external_eval")

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Checkpoints to evaluate
    checkpoints = {
        "arm0": "checkpoints/arm0_finetuned_best.pt",
        "arm1": "checkpoints/arm1_finetuned_best.pt",
        "arm2": "checkpoints/arm2_finetuned_best.pt",
        "arm3": "checkpoints/arm3_finetuned_best.pt"
    }
    
    results_dir = "outputs/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load splits.json to find test set for pretreat_m2b
    splits_path = "checkpoints/splits.json"
    test_cases = None
    if os.path.exists(splits_path):
        with open(splits_path, "r") as f:
            splits = json.load(f)
            test_cases = splits.get("test")
            logger.info(f"Loaded {len(test_cases)} test cases for pretreat_m2b from splits.json")
    else:
        logger.warning("No splits.json found. Will evaluate on all available pretreat_m2b cases.")
        
    # Cohorts to evaluate
    cohorts = {
        "pretreat_m2b_test": {
            "dataset_name": "pretreat_m2b",
            "cases": test_cases,
            "labeled": True
        },
        "brainmetshare": {
            "dataset_name": "brainmetshare",
            "cases": None,
            "labeled": True
        },
        "brainmetshare_test": {
            "dataset_name": "brainmetshare_test",
            "cases": None,
            "labeled": False
        },
        "ucsf_bmsr": {
            "dataset_name": "ucsf_bmsr",
            "cases": None,
            "labeled": True
        }
    }
    
    device = config.get("device", "cuda")
    
    # Loop over models and cohorts
    for model_name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint for {model_name} not found at {ckpt_path}. Skipping.")
            continue
            
        for cohort_name, cohort_info in cohorts.items():
            dataset_name = cohort_info["dataset_name"]
            
            # Check if dataset is preprocessed
            if not os.path.exists(os.path.join("data/preprocessed", dataset_name)):
                logger.warning(f"Dataset {dataset_name} has not been preprocessed. Skipping evaluation on {cohort_name}.")
                continue
                
            logger.info(f"Evaluating {model_name} on {cohort_name}...")
            
            try:
                summary, patient_wise = run_evaluation(
                    model_path=ckpt_path,
                    dataset_name=dataset_name,
                    allowed_cases=cohort_info["cases"],
                    device=device,
                    labeled=cohort_info["labeled"]
                )
                
                # Save summary metrics
                summary_path = os.path.join(results_dir, f"{model_name}_{cohort_name}_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=4)
                    
                # Save patient-wise metrics
                patient_path = os.path.join(results_dir, f"{model_name}_{cohort_name}_patients.json")
                with open(patient_path, "w") as f:
                    json.dump(patient_wise, f, indent=4)
                    
                logger.info(f"  Saved evaluation results to {summary_path}")
            except Exception as e:
                logger.error(f"  Error evaluating {model_name} on {cohort_name}: {str(e)}", exc_info=True)
                
    logger.info("External evaluations completed.")

if __name__ == "__main__":
    main()
