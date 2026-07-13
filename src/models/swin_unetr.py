"""
Swin UNETR model wrapper.
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRWrapper(nn.Module):
    """Swin UNETR model wrapper.
    
    A thin wrapper around monai.networks.nets.SwinUNETR.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Dictionary containing model hyperparameters:
                img_size: Tuple[int, int, int] (default (96, 96, 96))
                in_channels: int (default 4)
                out_channels: int (default 3)
                feature_size: int (default 48)
                use_checkpoint: bool (default True)
        """
        super().__init__()
        self.config = config
        
        self.model = SwinUNETR(
            img_size=tuple(config.get("img_size", (96, 96, 96))),
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("out_channels", 3),
            feature_size=config.get("feature_size", 48),
            use_checkpoint=config.get("use_checkpoint", True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W, D].
            
        Returns:
            Segmentation logits of shape [B, OutChannels, H, W, D].
        """
        return self.model(x)
        
    def get_encoder(self) -> nn.Module:
        """Returns the Swin ViT encoder backbone."""
        return self.model.swinViT
