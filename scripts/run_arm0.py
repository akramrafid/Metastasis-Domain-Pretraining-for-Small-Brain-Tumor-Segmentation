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
    finetune_epochs = config.get("finetune", {}).get("epochs", 0)
    if finetune_epochs > 0:
        print(f"\n>>> Starting Supervised Fine-tuning ({finetune_epochs} epochs, Sensitivity-Specificity tradeoff loss)...")
        train_finetune(config)
    else:
        print("\n>>> Skipping fine-tuning (epochs=0).")
        
    print("\n>>> Arm 0 pipeline complete.")

if __name__ == "__main__":
    main()
