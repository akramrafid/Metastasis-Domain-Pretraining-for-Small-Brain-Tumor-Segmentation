"""
Execution script for Arm 1 (baseline).
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from src.training.ssl_pretrain import train_ssl
from src.training.finetune import train_finetune

@hydra.main(config_path="../configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # Resolve config
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Apply experiment overrides if running directly
    # In Hydra, the config is automatically updated when running --config-name or overrides,
    # but we can explicitly log what we are running.
    print(f"==================================================")
    print(f"Running Arm 1 experiment: {config.get('name')}")
    print(f"==================================================")
    
    # Step 1: Self-supervised pretraining on BraTS-GLI
    print("\n>>> Step 1: Starting Self-Supervised Pretraining...")
    train_ssl(config)
    
    # Step 2: Supervised fine-tuning on Pretreat-MetsToBrain
    print("\n>>> Step 2: Starting Supervised Fine-tuning...")
    train_finetune(config)

if __name__ == "__main__":
    main()
