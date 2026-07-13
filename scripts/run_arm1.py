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
    
    print(f"==================================================")
    print(f"Running Arm 1 experiment: {config.get('name')}")
    print(f"==================================================")
    
    # Step 1: Self-supervised pretraining on BraTS-GLI
    pretrain_epochs = config.get("pretrain", {}).get("epochs", 0)
    if pretrain_epochs > 0:
        print(f"\n>>> Step 1: Starting Self-Supervised Pretraining ({pretrain_epochs} epochs)...")
        train_ssl(config)
    else:
        print("\n>>> Step 1: Skipping pretraining (epochs=0).")
    
    # Step 2: Supervised fine-tuning on Pretreat-MetsToBrain
    finetune_epochs = config.get("finetune", {}).get("epochs", 0)
    if finetune_epochs > 0:
        print(f"\n>>> Step 2: Starting Supervised Fine-tuning ({finetune_epochs} epochs)...")
        train_finetune(config)
    else:
        print("\n>>> Step 2: Skipping fine-tuning (epochs=0).")
    
    print("\n>>> Arm 1 pipeline complete.")

if __name__ == "__main__":
    main()
