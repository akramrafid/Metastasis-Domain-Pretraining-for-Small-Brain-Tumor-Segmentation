"""
Self-supervised pretraining heads (reconstruction + contrastive).
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SSLHeads(nn.Module):
    """Self-supervised pretraining model for Swin UNETR.
    
    Combines:
    1. Reconstruction: Decoder output of Swin UNETR (reconstructs 4-channel input).
    2. Contrastive: Global average pooling + MLP projection head on the encoder bottleneck.
    """
    def __init__(
        self,
        config: dict,
        projection_dim: int = 128
    ) -> None:
        """
        Args:
            config: Model configuration dictionary.
            projection_dim: Dimensionality of contrastive embedding space (default 128).
        """
        super().__init__()
        self.config = config
        
        # Reconstruction network is a SwinUNETR with out_channels = in_channels (e.g., 4)
        self.swin_unetr = SwinUNETR(  # type: ignore[call-arg]
            img_size=tuple(config.get("img_size", (96, 96, 96))),
            in_channels=config.get("in_channels", 4),
            out_channels=config.get("in_channels", 4), # Reconstruct all input channels
            feature_size=config.get("feature_size", 48),
            use_checkpoint=config.get("use_checkpoint", True)
        )
        
        # Encoder bottleneck dimension: feature_size * 8 * 2 (or feature_size * 16 depending on stage)
        # For feature_size = 48, bottleneck features has channel size 48 * 8 = 384 or 48 * 16 = 768.
        # In MONAI SwinViT, the final output feature dim is feature_size * 16 (for 48, it is 768).
        bottleneck_dim = config.get("feature_size", 48) * 16
        
        # Contrastive projection MLP head
        self.contrastive_projection = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass for pretraining.
        
        Args:
            x: Input volume of shape [B, C, H, W, D].
            
        Returns:
            Tuple of:
                - reconstructed_volume: [B, C, H, W, D]
                - contrastive_embedding: [B, projection_dim]
        """
        # Get encoder features. MONAI's SwinUNETR.swinViT returns hidden states
        hidden_states = self.swin_unetr.swinViT(x)
        # The final bottleneck feature is the last element of the hidden states list
        bottleneck = hidden_states[-1]
        
        # Run full decoder to get reconstruction
        reconstructed = self.swin_unetr(x)
        
        # Run contrastive projection on bottleneck
        contrastive_emb = self.contrastive_projection(bottleneck)
        # L2 normalize contrastive embeddings
        contrastive_emb = nn.functional.normalize(contrastive_emb, p=2, dim=1)
        
        return reconstructed, contrastive_emb
        
    def get_encoder_state_dict(self) -> dict:
        """Returns the state dict of the SwinViT encoder backbone."""
        return self.swin_unetr.swinViT.state_dict()
