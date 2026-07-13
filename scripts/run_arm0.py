"""
Execution script for Arm 0 (loss-based baseline).
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from src.training.finetune import train_finetune

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    
    print(f"==================================================")
    print(f"Running Arm 0 experiment: {config.get('name')}")
    print(f"==================================================")
    
    # Arm 0 uses the pretraining base of Arm 1, hence only supervised fine-tuning is run
    print("\n>>> Starting Supervised Fine-tuning (Sensitivity-Specificity tradeoff loss)...")
    train_finetune(config)

if __name__ == "__main__":
    main()
