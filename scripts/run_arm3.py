"""
Execution script for Arm 3 (combined: domain-matched + loss tradeoff).
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from src.training.finetune import train_finetune

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    
    print(f"==================================================")
    print(f"Running Arm 3 experiment: {config.get('name')}")
    print(f"==================================================")
    
    # Arm 3 uses the pretraining base from Arm 2, hence only supervised fine-tuning is run
    print("\n>>> Starting Supervised Fine-tuning (Sensitivity-Specificity tradeoff loss)...")
    train_finetune(config)

if __name__ == "__main__":
    main()
