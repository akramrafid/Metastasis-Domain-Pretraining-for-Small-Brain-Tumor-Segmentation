"""
Supervised fine-tuning loop for brain tumor segmentation.
Loads a pretrained encoder, attaches decoder, and trains on metastasis data.
"""
from typing import Dict, Any, List
import os
import random
import torch
import torch.optim as optim
import wandb
from monai.data import DataLoader as MonaiDataLoader
from src.data.datasets import BrainModalityDataset
from src.data.dataloader_factory import get_transforms
from src.models.swin_unetr import SwinUNETRWrapper
from src.training.losses import get_loss_function
from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logger
from src.utils.checkpoint import save_checkpoint

logger = setup_logger("finetune")

def get_deterministic_splits(cases: List[str], seed: int = 42) -> tuple:
    """Splits case list deterministically into train, val, and test splits (70/15/15)."""
    cases_sorted = sorted(cases)
    # Shuffle with a local random instance to preserve global seed state
    rng = random.Random(seed)
    rng.shuffle(cases_sorted)
    
    num_cases = len(cases_sorted)
    train_end = int(0.7 * num_cases)
    val_end = int(0.85 * num_cases)
    
    train_cases = cases_sorted[:train_end]
    val_cases = cases_sorted[train_end:val_end]
    test_cases = cases_sorted[val_end:]
    
    return train_cases, val_cases, test_cases

class SplitSubsetDataset(torch.utils.data.Dataset):
    """Wraps a BrainModalityDataset and filters it to a subset of cases."""
    def __init__(self, dataset: BrainModalityDataset, allowed_cases: List[str]) -> None:
        self.dataset = dataset
        self.allowed_cases = set(allowed_cases)
        
        # Filter indices
        self.indices = [
            i for i, case in enumerate(dataset.cases)
            if case in self.allowed_cases
        ]
        
    def __len__(self) -> int:
        return len(self.indices)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

def train_finetune(config: Dict[str, Any]) -> None:
    """Runs supervised fine-tuning.
    
    Args:
        config: Configuration dictionary.
    """
    set_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    preprocessed_dir = config.get("preprocessed_dir", "data/preprocessed")
    dataset_name = config["finetune"]["data"]
    
    # Get all available cases to perform split
    base_dataset = BrainModalityDataset(dataset_name=dataset_name, preprocessed_dir=preprocessed_dir)
    if len(base_dataset) == 0:
        raise FileNotFoundError(f"No preprocessed cases found for dataset {dataset_name} at {preprocessed_dir}")
        
    train_cases, val_cases, test_cases = get_deterministic_splits(base_dataset.cases, seed=config.get("seed", 42))
    logger.info(f"Split sizes: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)}")
    
    # Save the splits to a json file for documentation and statistical test evaluation
    splits_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "splits.json")
    os.makedirs(os.path.dirname(splits_path), exist_ok=True)
    with open(splits_path, "w") as f:
        import json
        json.dump({"train": train_cases, "val": val_cases, "test": test_cases}, f, indent=4)
        
    # Build datasets with appropriate splits
    patch_size = tuple(config["model"].get("img_size", (96, 96, 96)))
    train_ds = SplitSubsetDataset(
        dataset=BrainModalityDataset(dataset_name=dataset_name, preprocessed_dir=preprocessed_dir, transforms=get_transforms("train", patch_size)),
        allowed_cases=train_cases
    )
    val_ds = SplitSubsetDataset(
        dataset=BrainModalityDataset(dataset_name=dataset_name, preprocessed_dir=preprocessed_dir, transforms=get_transforms("val", patch_size)),
        allowed_cases=val_cases
    )
    
    train_loader = MonaiDataLoader(train_ds, batch_size=config["finetune"]["batch_size"], shuffle=True, num_workers=2)
    val_loader = MonaiDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    
    logger.info("Initializing Swin UNETR Wrapper...")
    model = SwinUNETRWrapper(config=config["model"])
    
    # Load pretrained encoder weights
    pretrained_path = config["finetune"].get("pretrained_path")
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained encoder weights from {pretrained_path}...")
        model.get_encoder().load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    else:
        logger.warning("No valid pretrained encoder path found. Training from scratch/random initialization.")
        
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["finetune"]["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["finetune"]["epochs"])
    
    # Setup loss
    loss_type = config["finetune"]["loss_type"]
    criterion = get_loss_function(loss_type, config["finetune"].get("loss_config", {}))
    
    # Initialize W&B logging if config specifies
    if config.get("wandb") and config["wandb"].get("mode") != "disabled":
        wandb.init(
            project=config["wandb"]["project"],
            name=f"finetune_{config['name']}",
            config=config,
            mode=config["wandb"].get("mode", "offline")
        )
        
    epochs = config["finetune"]["epochs"]
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    best_val_loss = float("inf")
    
    logger.info(f"Starting Fine-tuning for {epochs} epochs using {loss_type} loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        mean_loss = epoch_loss / len(train_loader)
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item()
                
        mean_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {mean_loss:.4f} - Val Loss: {mean_val_loss:.4f}")
        scheduler.step()
        
        if wandb.run:
            wandb.log({
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_loss": mean_val_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
        # Save best checkpoint
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            ckpt_path = os.path.join(checkpoint_dir, f"{config['name']}_finetuned_best.pt")
            logger.info(f"  New best validation loss! Saving model to {ckpt_path}...")
            save_checkpoint(
                state=model.state_dict(),
                filepath=ckpt_path
            )
            
    if wandb.run:
        wandb.finish()
    logger.info("Fine-tuning completed successfully!")
