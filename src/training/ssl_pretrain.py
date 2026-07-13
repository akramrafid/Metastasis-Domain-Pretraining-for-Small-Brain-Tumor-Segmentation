"""
Self-supervised pretraining loop using reconstruction and contrastive heads.
"""
from typing import Dict, Any, List
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.data.dataloader_factory import build_dataloader
from src.models.ssl_heads import SSLHeads
from src.utils.seed import set_seed
from src.utils.logging_utils import setup_logger
from src.utils.checkpoint import save_checkpoint

logger = setup_logger("ssl_pretrain")

def info_nce_loss(features_a: torch.Tensor, features_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Computes InfoNCE contrastive loss between two representations.
    
    Args:
        features_a: L2-normalized embeddings of view A [B, D].
        features_b: L2-normalized embeddings of view B [B, D].
        temperature: Softmax temperature parameter.
        
    Returns:
        Scalar InfoNCE loss.
    """
    batch_size = features_a.shape[0]
    # Cosine similarity matrix
    similarity_matrix = torch.matmul(features_a, features_b.T) / temperature
    labels = torch.arange(batch_size, device=features_a.device)
    
    loss_a = nn.functional.cross_entropy(similarity_matrix, labels)
    loss_b = nn.functional.cross_entropy(similarity_matrix.T, labels)
    return (loss_a + loss_b) / 2.0

def apply_random_masking(image: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor:
    """Applies random patch masking (setting to zero) for reconstruction pretext task.
    
    Args:
        image: Input tensor of shape [B, C, H, W, D].
        mask_ratio: Fraction of patches to mask out.
        
    Returns:
        Masked image tensor.
    """
    masked_image = image.clone()
    b, c, h, w, d = image.shape
    patch_size = 16  # 16x16x16 patches
    
    for batch_idx in range(b):
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                for k in range(0, d, patch_size):
                    if torch.rand(1).item() < mask_ratio:
                        # Mask out this block
                        masked_image[batch_idx, :, i:i+patch_size, j:j+patch_size, k:k+patch_size] = 0.0
    return masked_image

def train_ssl(config: Dict[str, Any]) -> None:
    """Runs the self-supervised pretraining loop.
    
    Args:
        config: Dictionary containing hyperparameters and configurations.
    """
    set_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    
    # Configure file logging dynamically
    log_dir = config.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{config['name']}_pretrain.log")
    
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
    
    # 1. Identify validation and test splits to exclude them from pretraining (prevent data leakage)
    val_cases: List[str] = []
    test_cases: List[str] = []
    
    if config["pretrain"]["data"] == "pretreat_m2b":
        splits_path = os.path.join(config.get("checkpoint_dir", "checkpoints"), "splits.json")
        if os.path.exists(splits_path):
            logger.info(f"Loading splits from {splits_path} to exclude val/test sets from pretraining...")
            with open(splits_path, "r") as f:
                splits = json.load(f)
            val_cases = splits.get("val", [])
            test_cases = splits.get("test", [])
        else:
            # Dynamically compute splits from the preprocessed cases if splits.json doesn't exist yet
            preprocessed_dir = config.get("preprocessed_dir", "data/preprocessed")
            dataset_dir = os.path.join(preprocessed_dir, "pretreat_m2b")
            if os.path.exists(dataset_dir):
                logger.info(f"Computing splits dynamically from {dataset_dir} to exclude val/test sets from pretraining...")
                all_cases = sorted(os.listdir(dataset_dir))
                from src.training.finetune import get_deterministic_splits
                _, val_cases, test_cases = get_deterministic_splits(all_cases, seed=config.get("seed", 42))
                
    exclude_list = val_cases + test_cases
    logger.info(f"Excluding {len(exclude_list)} val/test cases from pretraining...")
    
    logger.info("Building pretraining DataLoader...")
    dataloader = build_dataloader(
        dataset_name=config["pretrain"]["data"],
        split="train",
        batch_size=config["pretrain"]["batch_size"],
        num_workers=config.get("num_workers", 0),
        preprocessed_dir=config.get("preprocessed_dir", "data/preprocessed"),
        exclude_cases=exclude_list
    )
    
    # Hard assertion to prevent data leakage permanently
    exclude_set = set(exclude_list)
    loaded_cases = set(dataloader.dataset.cases)
    intersection = loaded_cases.intersection(exclude_set)
    assert len(intersection) == 0, f"CRITICAL: DATA LEAKAGE DETECTED! SSL pretraining contains fine-tuning val/test cases: {intersection}"
    
    logger.info("Initializing Swin UNETR and SSL Heads...")
    model = SSLHeads(config=config["model"])
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["pretrain"]["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["pretrain"]["epochs"])
    criterion_recon = nn.L1Loss()
    
    epochs = config["pretrain"]["epochs"]
    save_interval = config["pretrain"]["save_interval"]
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Checkpoint-and-resume setup
    start_epoch = 1
    latest_state_path = os.path.join(checkpoint_dir, f"{config['name']}_pretrain_latest.pt")
    if os.path.exists(latest_state_path):
        logger.info(f"Found latest pretraining checkpoint at {latest_state_path}. Resuming...")
        try:
            checkpoint = torch.load(latest_state_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Successfully resumed pretraining from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Could not load resume checkpoint: {str(e)}. Training from scratch.")
            
    # Initialize W&B logging if config specifies
    if config.get("wandb") and config["wandb"].get("mode") != "disabled":
        wandb.init(
            project=config["wandb"]["project"],
            name=f"pretrain_{config['name']}",
            config=config,
            mode=config["wandb"].get("mode", "offline"),
            resume="allow" if os.path.exists(latest_state_path) else None
        )
        
    # Setup mixed precision GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    grad_accum_steps = config["pretrain"].get("grad_accum_steps", 4)
    
    logger.info(f"Starting Pretraining from epoch {start_epoch} to {epochs} (Effective Batch Size: {config['pretrain']['batch_size'] * grad_accum_steps}) on {device}...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_recon_loss = 0.0
        epoch_contrast_loss = 0.0
        epoch_total_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device, dtype=torch.float32) # [B, 4, H, W, D]
            # Strip MONAI MetaTensor to plain Tensor to avoid __torch_function__
            # overhead inside SwinUNETR gradient checkpointing
            if hasattr(images, 'as_tensor'):
                images = images.as_tensor()
            
            # Generate two views with different maskings
            masked_a = apply_random_masking(images)
            masked_b = apply_random_masking(images)
            
            # Forward pass under autocast
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                recon_a, emb_a = model(masked_a)
                recon_b, emb_b = model(masked_b)
                
                # Reconstruction loss (against original images)
                loss_recon = (criterion_recon(recon_a, images) + criterion_recon(recon_b, images)) / 2.0
                
                # Contrastive InfoNCE loss
                loss_contrast = info_nce_loss(emb_a, emb_b)
                
                # Total loss scaled by gradient accumulation steps
                w_recon = config["pretrain"].get("recon_loss_weight", 1.0)
                w_contrast = config["pretrain"].get("contrast_loss_weight", 1.0)
                loss = w_recon * loss_recon + w_contrast * loss_contrast
                loss_scaled = loss / grad_accum_steps
                
            # Backward pass under scaled gradients
            scaler.scale(loss_scaled).backward()
            
            # Parameter update step
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            epoch_recon_loss += loss_recon.item()
            epoch_contrast_loss += loss_contrast.item()
            epoch_total_loss += loss.item()
            
            # Per-batch progress logging
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
                avg_loss = epoch_total_loss / (batch_idx + 1)
                print(f"\r  Epoch {epoch}/{epochs} - Batch {batch_idx + 1}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}", end="", flush=True)
            
        print()  # newline after batch progress
        # Logging
        mean_recon = epoch_recon_loss / len(dataloader)
        mean_contrast = epoch_contrast_loss / len(dataloader)
        mean_total = epoch_total_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {mean_total:.4f} (Recon: {mean_recon:.4f}, Contrast: {mean_contrast:.4f})")
        scheduler.step()
        
        if wandb.run:
            wandb.log({
                "epoch": epoch,
                "recon_loss": mean_recon,
                "contrast_loss": mean_contrast,
                "total_loss": mean_total,
                "lr": optimizer.param_groups[0]["lr"]
            })
            
        # Save checkpoints
        if epoch % save_interval == 0 or epoch == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"{config['name']}_encoder_ep{epoch}.pt")
            logger.info(f"Saving encoder checkpoint to {ckpt_path}...")
            save_checkpoint(
                state=model.get_encoder_state_dict(),
                filepath=ckpt_path
            )
            
        # Save latest full state for resume
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config
        }
        save_checkpoint(latest_state, latest_state_path)
        
    if wandb.run:
        wandb.finish()
    logger.info("Pretraining completed successfully!")
