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
from monai.inferers import sliding_window_inference
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
    
    # Configure file logging dynamically
    log_dir = config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['name']}_finetune.log")
    
    import logging
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not has_file_handler:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
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
    epochs = config["finetune"]["epochs"]
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    # Checkpoint-and-resume setup
    start_epoch = 1
    latest_state_path = os.path.join(checkpoint_dir, f"{config['name']}_finetune_latest.pt")
    if os.path.exists(latest_state_path):
        logger.info(f"Found latest fine-tuning checkpoint at {latest_state_path}. Resuming...")
        try:
            checkpoint = torch.load(latest_state_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info(f"Successfully resumed fine-tuning from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            logger.warning(f"Could not load resume checkpoint: {str(e)}. Training from scratch.")
            
    # Initialize W&B logging if config specifies
    if config.get("wandb") and config["wandb"].get("mode") != "disabled":
        wandb.init(
            project=config["wandb"]["project"],
            name=f"finetune_{config['name']}",
            config=config,
            mode=config["wandb"].get("mode", "offline"),
            resume="allow" if os.path.exists(latest_state_path) else None
        )
        
    # Setup mixed precision GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    grad_accum_steps = config["finetune"].get("grad_accum_steps", 4)
    
    logger.info(f"Starting Fine-tuning from epoch {start_epoch} to {epochs} using {loss_type} loss (Effective Batch Size: {config['finetune']['batch_size'] * grad_accum_steps}) on {device}...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device, dtype=torch.float32)
            labels = batch["label"].to(device, dtype=torch.float32)
            # Strip MONAI MetaTensor to plain Tensor to avoid __torch_function__
            # overhead inside SwinUNETR gradient checkpointing
            if hasattr(images, 'as_tensor'):
                images = images.as_tensor()
            if hasattr(labels, 'as_tensor'):
                labels = labels.as_tensor()
            
            # Forward pass under autocast
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_scaled = loss / grad_accum_steps
                
            # Backward pass under scaled gradients
            scaler.scale(loss_scaled).backward()
            
            # Parameter update step
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_loss += loss.item()
            
            # Per-batch progress logging
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"\r  Epoch {epoch}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}", end="", flush=True)
            
        print()  # newline after batch progress
        mean_loss = epoch_loss / len(train_loader)
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device, dtype=torch.float32)
                val_labels = val_batch["label"].to(device, dtype=torch.float32)
                # Strip MONAI MetaTensor
                if hasattr(val_images, 'as_tensor'):
                    val_images = val_images.as_tensor()
                if hasattr(val_labels, 'as_tensor'):
                    val_labels = val_labels.as_tensor()
                
                # Autocast validation forward pass
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    val_outputs = sliding_window_inference(
                        inputs=val_images,
                        roi_size=patch_size,
                        sw_batch_size=4,
                        predictor=model,
                        overlap=0.25,
                        device=device
                    )
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
            
        # Save latest full state for resume
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config
        }
        save_checkpoint(latest_state, latest_state_path)
        
    if wandb.run:
        wandb.finish()
    logger.info("Fine-tuning completed successfully!")
